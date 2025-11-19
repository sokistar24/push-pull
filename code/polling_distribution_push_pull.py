"""
polling_distribution_push_pull_pct.py

Run push (L-SIP) and pull (Whittle-like) once on synthetic_data.csv,
and plot the distribution of:

- Push: successful transmissions per category
- Pull: polls per category

The y-axis is a percentage of the total number of samples
(T * number_of_nodes), not raw packet counts.

Assumes directory structure:

project_root/
    code/
        polling_distribution_push_pull_pct.py
    data/
        synthetic_data.csv
    results/
        (outputs will be saved here)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import CubicHermiteSpline

# =======================
# Configuration
# =======================
N_STEPS      = 10000
M_FIXED      = 5

# Push (L-SIP) params
ALPHA   = 0.9
BETA    = 0.01
EPSILON = 2.5
DT      = 1.0
CW_MIN  = 4
CW_MAX  = 256
RNG     = np.random.default_rng(1234)

# Optional staggered start for PUSH
STAGGER_INIT         = True
INIT_STAGGER_SPREAD  = 8
STAGGER_MODE         = "random"

# Pull (Whittle-like) params
AOII_PENALTY  = 0.5
INITIAL_VALUE = 20.0
INITIAL_RATE  = 0.01
BETA_1        = 0.9
BETA_2        = 0.9


# =======================
# Helpers (ID → Category)
# =======================
def extract_node_id(col_name):
    """Extract numeric node ID from column name."""
    digits = ''.join(filter(str.isdigit, str(col_name)))
    return int(digits) if digits else None


def categorize_node(node_id):
    """Categorize node based on ID."""
    if node_id is None:
        return "Other"
    if 1 <= node_id <= 10:
        return "Category A"
    if 11 <= node_id <= 20:
        return "Category B"
    if 21 <= node_id <= 30:
        return "Category C"
    return "Other"


def build_push_node_id_map(columns):
    """
    Map push nid (0-based enumerate over columns) → true numeric id (from column name) or fallback nid+1.
    """
    nid_map = {}
    for nid, c in enumerate(columns):
        cid = extract_node_id(c)
        nid_map[nid] = cid if cid is not None else (nid + 1)
    return nid_map


# =======================
# Hermite helpers (needed for reconstruction)
# =======================
def reconstruct_from_transmissions(time_steps, transmissions, *, edge_mode="nan"):
    """
    Hermite reconstruction from (time, x1, x2) knots (push uses x1,x2; pull stores x1=z).
    No extrapolation: values outside [first_knot, last_knot] are NaN by default.
    """
    k_t, k_v, k_r = [], [], []
    for tx in transmissions:
        if tx is not None:
            k_t.append(tx["time"])
            k_v.append(tx["x1"])
            k_r.append(tx["x2"])

    t_all = np.asarray(time_steps, dtype=float)

    if len(k_t) == 0:
        return np.full_like(t_all, np.nan, dtype=float)
    if len(k_t) == 1:
        out = np.full_like(t_all, np.nan, dtype=float)
        out[t_all == float(k_t[0])] = float(k_v[0])
        return out

    times  = np.asarray(k_t, dtype=float)
    values = np.asarray(k_v, dtype=float)
    rates  = np.asarray(k_r, dtype=float)
    if rates.size != times.size:
        if rates.size == 0:
            rates = np.zeros_like(times)
        elif rates.size < times.size:
            rates = np.pad(rates, (0, times.size - rates.size), mode="edge")
        else:
            rates = rates[:times.size]

    cs = CubicHermiteSpline(times, values, rates, extrapolate=False)
    return cs(t_all)


# =======================
# PUSH (L-SIP) with channels + BEB
# =======================
class NodeLSIP:
    """dEWMA at edge + binary exponential backoff; transmits (x1, x2)."""
    def __init__(self, alpha=ALPHA, beta=BETA, epsilon=EPSILON,
                 cw_min=CW_MIN, cw_max=CW_MAX):
        self.alpha   = alpha
        self.beta    = beta
        self.epsilon = epsilon

        self.x1 = None
        self.x2 = None

        # last sink-known state (after success)
        self.x1_tx = None
        self.x2_tx = None
        self.t_tx  = None

        # contention
        self.cw_min = int(cw_min)
        self.cw_max = int(cw_max)
        self.cw = int(cw_min)
        self.backoff = 0

        self.init = False

    def update_dewma(self, z):
        z = float(z)
        if not self.init:
            self.x1 = z
            self.x2 = 0.0
            self.init = True
            return
        x1p   = self.x1 + self.x2 * DT
        self.x1 = self.alpha * z + (1 - self.alpha) * x1p
        self.x2 = self.beta  * (self.x1 - x1p) / DT + (1 - self.beta) * self.x2

    def want_tx(self, t):
        if self.t_tx is None:
            return True
        pred_sink = self.x1_tx + self.x2_tx * (t - self.t_tx)
        return abs(self.x1 - pred_sink) > self.epsilon

    def dec_backoff(self):
        if self.backoff > 0:
            self.backoff -= 1

    def pick_channel(self, M):
        return int(RNG.integers(0, M))

    def on_success(self, t):
        self.x1_tx = self.x1
        self.x2_tx = self.x2
        self.t_tx  = t
        self.cw = self.cw_min
        self.backoff = 0

    def on_collision(self):
        self.cw = min(self.cw * 2, self.cw_max)
        self.backoff = int(RNG.integers(0, self.cw))


def run_push_simulation(long_df, M_channels, *, epsilon=EPSILON):
    """
    Inputs: long_df with columns [timestamp, node_id, value]
    Returns per-node transmissions and global attempt/success/collision counts,
    plus per-node attempt/success tallies.
    """
    cols = long_df.columns.tolist()
    tcol = next((c for c in ["timestamp", "time", "t"] if c in cols), cols[0])
    ncol = next((c for c in ["node_id", "node", "id"] if c in cols), None)
    vcol = next(
        (c for c in ["value", "measurement", "data"] if c in cols),
        [c for c in cols if c not in [tcol, ncol]][0]
    )

    node_ids   = sorted(long_df[ncol].unique()) if ncol else [0]
    time_steps = sorted(long_df[tcol].unique())

    nodes = {nid: NodeLSIP(epsilon=epsilon) for nid in node_ids}

    # Staggered start (optional)
    if STAGGER_INIT and INIT_STAGGER_SPREAD > 0:
        for nid in node_ids:
            if STAGGER_MODE == "by_id":
                nodes[nid].backoff = int(nid % INIT_STAGGER_SPREAD)
            else:
                nodes[nid].backoff = int(RNG.integers(0, INIT_STAGGER_SPREAD))

    results = {
        "time_steps": time_steps,
        "measurements": {nid: [] for nid in node_ids},
        "x1_estimates": {nid: [] for nid in node_ids},
        "x2_estimates": {nid: [] for nid in node_ids},
        "transmissions": {nid: [] for nid in node_ids},
        "total_attempts": 0,
        "total_success":  0,
        "total_collisions": 0,
        "attempts_per_node":  {nid: 0 for nid in node_ids},
        "successes_per_node": {nid: 0 for nid in node_ids},
    }

    grouped = long_df.groupby(tcol)

    for t, group in grouped:
        # 1) Measurement updates
        for _, row in group.iterrows():
            nid = int(row[ncol]) if ncol else 0
            z   = float(row[vcol])
            nodes[nid].update_dewma(z)
            results["measurements"][nid].append(z)
            results["x1_estimates"][nid].append(nodes[nid].x1)
            results["x2_estimates"][nid].append(nodes[nid].x2)

        # 2) Decide attempts and channels
        attempts_by_ch = {ch: [] for ch in range(M_channels)}
        for nid in node_ids:
            nodes[nid].dec_backoff()
            if nodes[nid].backoff == 0 and nodes[nid].want_tx(t):
                ch = nodes[nid].pick_channel(M_channels)
                attempts_by_ch[ch].append(nid)
                results["attempts_per_node"][nid] += 1

        # 3) Resolve contention per channel
        tx_slot = {nid: None for nid in node_ids}
        for ch, contenders in attempts_by_ch.items():
            if len(contenders) == 1:
                nid = contenders[0]
                nodes[nid].on_success(t)
                tx_slot[nid] = {"time": t, "x1": nodes[nid].x1, "x2": nodes[nid].x2}
                results["total_attempts"] += 1
                results["total_success"]  += 1
                results["successes_per_node"][nid] += 1
            elif len(contenders) > 1:
                for nid in contenders:
                    nodes[nid].on_collision()
                results["total_attempts"]  += len(contenders)
                results["total_collisions"] += len(contenders)

        for nid in node_ids:
            results["transmissions"][nid].append(tx_slot[nid])

    T = len(time_steps)
    results["total_points"] = T * len(node_ids)
    return results


# =======================
# PULL (Whittle-like)
# =======================
def dewma_update_pull(z, x1, x2, dt, beta1=BETA_1, beta2=BETA_2):
    dt = max(1.0, float(dt))
    x1p = x1 + x2 * dt
    x1n = beta1 * z + (1 - beta1) * x1p
    x2n = beta2 * (x1n - x1p) / dt + (1 - beta2) * x2
    return float(x1n), float(x2n)


def run_pull_simulation(pivot_df, columns, M, *, aoii_penalty=AOII_PENALTY):
    """
    Whittle-like index: |(Δt+1)*x2| - aoii_penalty.
    Poll nodes with index>=0 (top M if too many).
    TX knot at each poll: (time, x1=z, x2=updated slope).
    """
    T = len(pivot_df)
    x1 = {c: INITIAL_VALUE for c in columns}
    x2 = {c: INITIAL_RATE for c in columns}
    last_t = {c: 0 for c in columns}

    results = {
        "time_steps": list(range(T)),
        "measurements": {c: [] for c in columns},
        "transmissions": {c: [] for c in columns},
        "total_polls": 0,
        "total_points": T * len(columns),
        "polls_per_node": {c: 0 for c in columns},
    }

    for t in range(T):
        for c in columns:
            results["measurements"][c].append(float(pivot_df.loc[t, c]))

        idx = {}
        for c in columns:
            dt = t - last_t[c]
            future_passive = abs((dt + 1) * x2[c])
            idx[c] = future_passive - aoii_penalty

        cand = [c for c in columns if idx[c] >= 0]
        if len(cand) > M:
            cand = sorted(cand, key=lambda cc: idx[cc], reverse=True)[:M]

        tx_slot = {c: None for c in columns}
        for c in cand:
            z  = float(pivot_df.loc[t, c])
            dt = t - last_t[c]
            x1[c], x2[c] = dewma_update_pull(z, x1[c], x2[c], dt)
            last_t[c] = t
            tx_slot[c] = {"time": t, "x1": z, "x2": x2[c]}
            results["total_polls"] += 1
            results["polls_per_node"][c] += 1

        for c in columns:
            results["transmissions"][c].append(tx_slot[c])

    return results


# =======================
# Push/Pull category summaries
# =======================
def summarize_push_success_by_category(push_res, nid_map):
    """
    Returns DataFrame with columns: ['category','successes'] aggregated by A/B/C/Other.
    Uses only successful transmissions (delivered to sink).
    """
    rows = []
    for nid, succ in push_res["successes_per_node"].items():
        numeric_id = nid_map.get(nid, nid + 1)
        cat = categorize_node(numeric_id)
        rows.append((cat, succ))
    df = pd.DataFrame(rows, columns=["category", "successes"])
    return df.groupby("category", as_index=False)["successes"].sum()


def summarize_pull_polls_by_category(pull_res, columns):
    """
    Returns DataFrame with columns: ['category','polls'] aggregated by A/B/C/Other.
    """
    rows = []
    for c in columns:
        cnt = pull_res["polls_per_node"][c]
        numeric_id = extract_node_id(c)
        cat = categorize_node(numeric_id if numeric_id is not None else None)
        rows.append((cat, cnt))
    df = pd.DataFrame(rows, columns=["category", "polls"])
    return df.groupby("category", as_index=False)["polls"].sum()


# =======================
# Plot: distribution as percentage of total samples
# =======================
def plot_polling_distribution_percentage(push_succ_df,
                                         pull_cat_df,
                                         total_samples: int,
                                         results_dir: Path,
                                         filename: str = "polling_distribution_push_pull_pct.png"):
    """
    Bar chart comparing Push successes vs Pull polls per Category (A/B/C),
    with y-axis as percentage of total samples.
    """
    cats = ["Category A", "Category B", "Category C"]
    push_vals = {row["category"]: row["successes"] for _, row in push_succ_df.iterrows()}
    pull_vals = {row["category"]: row["polls"]     for _, row in pull_cat_df.iterrows()}

    push_counts = [push_vals.get(c, 0) for c in cats]
    pull_counts = [pull_vals.get(c, 0) for c in cats]

    # Convert counts to percentages of total samples
    if total_samples <= 0:
        raise ValueError("total_samples must be positive to compute percentages.")

    push_pct = [100.0 * c / total_samples for c in push_counts]
    pull_pct = [100.0 * c / total_samples for c in pull_counts]

    x = np.arange(len(cats))
    w = 0.38

    results_dir.mkdir(exist_ok=True)
    savepath = results_dir / filename

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, push_pct, width=w, label="Push")
    ax.bar(x + w / 2, pull_pct, width=w, label="Pull")

    # Ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=14)
    ax.tick_params(axis="both", labelsize=14)

    ax.set_xlabel("Sensor category", fontsize=14)
    ax.set_ylabel("Percentage of total samples (%)", fontsize=14)

    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=14)

    # Annotate bars with percentages
    for xi, y in zip(x - w / 2, push_pct):
        ax.text(xi, y, f"{y:.1f}%", ha="center", va="bottom", fontsize=12)
    for xi, y in zip(x + w / 2, pull_pct):
        ax.text(xi, y, f"{y:.1f}%", ha="center", va="bottom", fontsize=12)

    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Saved polling distribution figure (percentage) to: {savepath}")
    plt.close(fig)


# =======================
# Main: run once and plot distribution
# =======================
def run_distribution_plot_only(dataset_path: Path, results_dir: Path):
    print("Running push and pull once, plotting distribution of transmitted packets by category")
    print("Y-axis is percentage of total samples.")
    print("=" * 100)
    print(f"Dataset: {dataset_path}")
    print(f"Results directory: {results_dir}")
    print("=" * 100)

    # Load dataset wide-form
    pivot_df = pd.read_csv(dataset_path)
    pivot_df = pivot_df.apply(lambda x: x.fillna(x.mean()), axis=0).head(N_STEPS)
    columns  = [c for c in pivot_df.columns if c != "SN"]
    if not columns:
        columns = list(pivot_df.columns)

    # Build long-form for push and nid → numeric id map
    rows = []
    for t in range(len(pivot_df)):
        for nid, c in enumerate(columns):
            rows.append({"timestamp": t, "node_id": nid, "value": float(pivot_df.loc[t, c])})
    long_df = pd.DataFrame(rows)
    nid_map = build_push_node_id_map(columns)

    # Run simulations
    push_res = run_push_simulation(long_df, M_channels=M_FIXED, epsilon=EPSILON)
    pull_res = run_pull_simulation(pivot_df, columns, M_FIXED, aoii_penalty=AOII_PENALTY)

    total_samples = push_res["total_points"]  # T * num_nodes

    # Summaries
    push_cat_success = summarize_push_success_by_category(push_res, nid_map)
    pull_cat         = summarize_pull_polls_by_category(pull_res, columns)

    # Save summaries to CSV in results folder
    results_dir.mkdir(exist_ok=True)
    push_csv = results_dir / "push_success_by_category.csv"
    pull_csv = results_dir / "pull_polls_by_category.csv"
    push_cat_success.to_csv(push_csv, index=False)
    pull_cat.to_csv(pull_csv, index=False)

    print("\nPush successes by category (counts):")
    print(push_cat_success)
    print("\nPull polls by category (counts):")
    print(pull_cat)
    print(f"\nSaved category tables to:\n  {push_csv}\n  {pull_csv}")

    # Plot distribution as percentage of total samples
    plot_polling_distribution_percentage(
        push_cat_success,
        pull_cat,
        total_samples=total_samples,
        results_dir=results_dir,
        filename="polling_distribution_push_pull_pct.png",
    )


def main():
    # Assume script is in project_root/code
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    dataset_path = project_root / "data" / "synthetic_data.csv"
    results_dir  = project_root / "results"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    run_distribution_plot_only(dataset_path, results_dir)


if __name__ == "__main__":
    main()
