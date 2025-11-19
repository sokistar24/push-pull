"""
push_pull_param_sweep.py

Run push (L-SIP) vs pull (Whittle-like) param sweep and:
- Print metrics to console
- Save summary CSV/TXT into ../results
- Save sample reconstruction figure into ../results
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
from sklearn.metrics import mean_squared_error

# =======================
# Configuration
# =======================
N_STEPS      = 2000
M_FIXED      = 5                # keep M constant at 5
PARAM_VALUES = [0.1, 0.5, 1.0]  # sweep over ε (push) and AoII penalty (pull)

# Push (L-SIP) params
ALPHA   = 0.9
BETA    = 0.01
EPSILON = 1.0
DT      = 1.0
CW_MIN  = 4
CW_MAX  = 256
RNG     = np.random.default_rng(1234)

# Optional staggered start for PUSH
STAGGER_INIT         = True       # set False to disable
INIT_STAGGER_SPREAD  = 8          # initial backoff window (slots)
STAGGER_MODE         = "random"   # "random" or "by_id"

# Pull (Whittle-like) params
AOII_PENALTY  = 1.0
INITIAL_VALUE = 20.0
INITIAL_RATE  = 0.01
BETA_1        = 0.9
BETA_2        = 0.01


# =======================
# Hermite helpers + pooled metrics
# =======================
def hermite_from_value_rate(T, times, values, rates):
    t_all  = np.arange(T, dtype=float)
    times  = np.asarray(times,  dtype=float)
    values = np.asarray(values, dtype=float)
    rates  = np.asarray(rates,  dtype=float)

    if times.size == 0:
        return np.full(T, np.nan)
    if times.size == 1:
        v0, t0 = values[0], times[0]
        r0 = rates[0] if rates.size else 0.0
        return v0 + r0 * (t_all - t0)

    if rates.size != times.size:
        if rates.size == 0:
            rates = np.zeros_like(times)
        elif rates.size < times.size:
            rates = np.pad(rates, (0, times.size - rates.size), mode="edge")
        else:
            rates = rates[:times.size]

    cs = CubicHermiteSpline(times, values, rates, extrapolate=True)
    return cs(t_all)


def pooled_mse(true_list, recon_list):
    se = 0.0
    n  = 0
    for true, recon in zip(true_list, recon_list):
        a = np.asarray(true, float)
        b = np.asarray(recon, float)
        mask = np.isfinite(a) & np.isfinite(b)
        if not np.any(mask):
            continue
        diff = a[mask] - b[mask]
        se  += float(np.dot(diff, diff))
        n   += diff.size
    return se / n if n > 0 else np.nan


def pooled_rmse(true_list, recon_list):
    m = pooled_mse(true_list, recon_list)
    return np.sqrt(m) if np.isfinite(m) else np.nan


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
    Returns per-node measurements, x1_estimates, transmissions (success only),
    and global attempt/success/collision counts.
    """
    cols = long_df.columns.tolist()
    tcol = next((c for c in ['timestamp', 'time', 't'] if c in cols), cols[0])
    ncol = next((c for c in ['node_id', 'node', 'id'] if c in cols), None)
    vcol = next(
        (c for c in ['value', 'measurement', 'data'] if c in cols),
        [c for c in cols if c not in [tcol, ncol]][0]
    )

    node_ids   = sorted(long_df[ncol].unique()) if ncol else [0]
    time_steps = sorted(long_df[tcol].unique())
    T = len(time_steps)

    # create nodes with the chosen epsilon
    nodes = {nid: NodeLSIP(epsilon=epsilon) for nid in node_ids}

    # Staggered start (optional)
    if STAGGER_INIT and INIT_STAGGER_SPREAD > 0:
        for nid in node_ids:
            if STAGGER_MODE == "by_id":
                nodes[nid].backoff = int(nid % INIT_STAGGER_SPREAD)
            else:  # random
                nodes[nid].backoff = int(RNG.integers(0, INIT_STAGGER_SPREAD))

    results = {
        'time_steps': time_steps,
        'measurements': {nid: [] for nid in node_ids},
        'x1_estimates': {nid: [] for nid in node_ids},
        'x2_estimates': {nid: [] for nid in node_ids},
        'transmissions': {nid: [] for nid in node_ids},  # success only; else None
        'total_attempts': 0,
        'total_success':  0,
        'total_collisions': 0
    }

    grouped = long_df.groupby(tcol)

    for t, group in grouped:
        # 1) Filter updates
        for _, row in group.iterrows():
            nid = int(row[ncol]) if ncol else 0
            z   = float(row[vcol])
            nodes[nid].update_dewma(z)
            results['measurements'][nid].append(z)
            results['x1_estimates'][nid].append(nodes[nid].x1)
            results['x2_estimates'][nid].append(nodes[nid].x2)

        # 2) Decide attempts & channels
        attempts_by_ch = {ch: [] for ch in range(M_channels)}
        for nid in node_ids:
            nodes[nid].dec_backoff()
            if nodes[nid].backoff == 0 and nodes[nid].want_tx(t):
                ch = nodes[nid].pick_channel(M_channels)
                attempts_by_ch[ch].append(nid)

        # 3) Resolve per-channel
        tx_slot = {nid: None for nid in node_ids}
        for ch, contenders in attempts_by_ch.items():
            if len(contenders) == 1:
                nid = contenders[0]
                nodes[nid].on_success(t)
                # TX payload: (x1, x2)
                tx_slot[nid] = {'time': t, 'x1': nodes[nid].x1, 'x2': nodes[nid].x2}
                results['total_attempts'] += 1
                results['total_success']  += 1
            elif len(contenders) > 1:
                for nid in contenders:
                    nodes[nid].on_collision()
                results['total_attempts']  += len(contenders)
                results['total_collisions'] += len(contenders)

        for nid in node_ids:
            results['transmissions'][nid].append(tx_slot[nid])

    # add derived totals
    results['total_points'] = T * len(node_ids)
    return results


# =======================
# PULL (Whittle-like)
# =======================
def calculate_aoii_sink(current_time, last_received_time, last_rate_of_change):
    return abs((current_time - last_received_time) * last_rate_of_change)


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
        'time_steps': list(range(T)),
        'measurements': {c: [] for c in columns},
        'transmissions': {c: [] for c in columns},
        'total_polls': 0,
        'total_points': T * len(columns)
    }

    for t in range(T):
        for c in columns:
            results['measurements'][c].append(float(pivot_df.loc[t, c]))

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
            tx_slot[c] = {'time': t, 'x1': z, 'x2': x2[c]}
            results['total_polls'] += 1

        for c in columns:
            results['transmissions'][c].append(tx_slot[c])

    return results


# =======================
# Metrics
# =======================
def reconstruct_from_transmissions(time_steps, transmissions, *, edge_mode='nan'):
    """
    Hermite reconstruction from (time, x1, x2) knots (push uses x1,x2; pull stores x1=z).
    No extrapolation: values outside [first_knot, last_knot] are NaN by default.
    """
    k_t, k_v, k_r = [], [], []
    for tx in transmissions:
        if tx is not None:
            k_t.append(tx['time'])
            k_v.append(tx['x1'])
            k_r.append(tx['x2'])

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
            rates = np.pad(rates, (0, times.size - rates.size), mode='edge')
        else:
            rates = rates[:times.size]

    cs = CubicHermiteSpline(times, values, rates, extrapolate=False)
    return cs(t_all)  # NaN outside the span


def calculate_push_metrics(push_res):
    t_all = np.asarray(push_res['time_steps'], dtype=float)
    node_ids = list(push_res['measurements'].keys())

    # RMSE(z - x1) at success times only
    errs = []
    for nid in node_ids:
        z_series  = np.asarray(push_res['measurements'][nid], dtype=float)
        x1_series = np.asarray(push_res['x1_estimates'][nid], dtype=float)
        for i, tx in enumerate(push_res['transmissions'][nid]):
            if tx is not None:
                errs.append(z_series[i] - x1_series[i])
    transmission_rmse = float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan

    # Pooled recon RMSE vs z
    true_all, recon_all = [], []
    total_success = 0
    for nid in node_ids:
        recon = reconstruct_from_transmissions(t_all, push_res['transmissions'][nid], edge_mode='nan')
        true  = np.asarray(push_res['measurements'][nid], dtype=float)
        true_all.append(true)
        recon_all.append(recon)
        total_success += sum(1 for tx in push_res['transmissions'][nid] if tx is not None)
    recon_rmse = pooled_rmse(true_all, recon_all)

    total_points = push_res['total_points']
    tx_rate      = total_success / total_points if total_points > 0 else 0.0
    compression  = 1.0 - tx_rate

    attempts   = push_res['total_attempts']
    successes  = push_res['total_success']
    collisions = push_res['total_collisions']
    succ_given_attempt = successes / attempts if attempts > 0 else np.nan
    coll_given_attempt = collisions / attempts if attempts > 0 else np.nan

    return {
        'transmission_rmse': transmission_rmse,
        'recon_rmse': recon_rmse,
        'tx_rate': tx_rate,
        'compression': compression,
        'attempts': attempts,
        'successes': successes,
        'collisions': collisions,
        'succ_given_attempt': succ_given_attempt,
        'coll_given_attempt': coll_given_attempt
    }


def calculate_pull_metrics(pull_res):
    t_all = np.asarray(pull_res['time_steps'], dtype=float)
    cols  = list(pull_res['measurements'].keys())

    true_all, recon_all = [], []
    total_polls = 0
    for c in cols:
        recon = reconstruct_from_transmissions(t_all, pull_res['transmissions'][c])
        true  = np.asarray(pull_res['measurements'][c], dtype=float)
        true_all.append(true)
        recon_all.append(recon)
        total_polls += sum(1 for tx in pull_res['transmissions'][c] if tx is not None)

    recon_rmse = pooled_rmse(true_all, recon_all)

    total_points = pull_res['total_points']
    tx_rate     = total_polls / total_points if total_points > 0 else 0.0
    compression = 1.0 - tx_rate

    return {
        'recon_rmse': recon_rmse,
        'tx_rate': tx_rate,
        'compression': compression,
        'total_polls': total_polls
    }


# =======================
# Plot helper
# =======================
def plot_sample_reconstruction(pivot_df, push_res, pull_res, node_idx, results_dir: Path,
                               filename="sample_reconstruction_param0_5.png"):
    """
    Show reconstructions for one node (index) for push & pull with a single top legend,
    and save the figure into the results directory.
    """
    import matplotlib.lines as mlines

    results_dir.mkdir(exist_ok=True)
    savepath = results_dir / filename

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ---------- PUSH ----------
    t_all = push_res['time_steps']
    z_push = np.asarray(push_res['measurements'][node_idx], dtype=float)
    r_push = reconstruct_from_transmissions(
        t_all,
        push_res['transmissions'][node_idx],
        edge_mode='nan'
    )
    tx_idx = [i for i, tx in enumerate(push_res['transmissions'][node_idx]) if tx is not None]

    ax1.plot(t_all, z_push, color='tab:green', linewidth=2.2)
    ax1.plot(t_all, r_push, color='tab:blue',  linewidth=2.5)
    if tx_idx:
        ax1.scatter(np.array(t_all)[tx_idx], z_push[tx_idx], s=22, c='tab:red', zorder=3)

    mask_push = np.isfinite(r_push)
    rmse_push = float(
        np.sqrt(mean_squared_error(z_push[mask_push], r_push[mask_push]))
    ) if np.any(mask_push) else np.nan
    ax1.text(
        0.01, 0.98,
        f"Push (RMSE={rmse_push:.2f})",
        transform=ax1.transAxes,
        va='top', ha='left', fontsize=14
    )

    ax1.set_ylabel("Value", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)

    # ---------- PULL ----------
    cols = list(pull_res['measurements'].keys())
    col  = cols[node_idx % len(cols)]
    t_all_pull = pull_res['time_steps']
    z_pull = np.asarray(pull_res['measurements'][col], dtype=float)
    r_pull = reconstruct_from_transmissions(t_all_pull, pull_res['transmissions'][col])
    tx_idx_p = [i for i, tx in enumerate(pull_res['transmissions'][col]) if tx is not None]

    ax2.plot(t_all_pull, z_pull, color='tab:green', linewidth=2.2)
    ax2.plot(t_all_pull, r_pull, color='tab:blue',  linewidth=2.5)
    if tx_idx_p:
        ax2.scatter(np.array(t_all_pull)[tx_idx_p], z_pull[tx_idx_p], s=22, c='tab:red', zorder=3)

    mask_pull = np.isfinite(r_pull)
    rmse_pull = float(
        np.sqrt(mean_squared_error(z_pull[mask_pull], r_pull[mask_pull]))
    ) if np.any(mask_pull) else np.nan
    ax2.text(
        0.01, 0.98,
        f"Pull (RMSE={rmse_pull:.2f})",
        transform=ax2.transAxes,
        va='top', ha='left', fontsize=14
    )

    ax2.set_xlabel("Time step", fontsize=14)
    ax2.set_ylabel("Value", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)

    # ---------- LEGEND ----------
    handles = [
        mlines.Line2D([], [], color='tab:green', linewidth=2.5, label='Actual'),
        mlines.Line2D([], [], color='tab:blue',  linewidth=2.8, label='Reconstruction'),
        mlines.Line2D([], [], color='tab:red', marker='o', linestyle='None', markersize=6, label='TX'),
    ]
    fig.legend(
        handles=handles,
        loc='upper center',
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    print(f"Saved sample reconstruction figure to: {savepath}")
    plt.close(fig)


# =======================
# Main: parameter sweep with M fixed at 5
# =======================
def run_param_sweep(dataset_path: Path, results_dir: Path):
    print("Param sweep with M=5 — push uses ε, pull uses AoII penalty; no extrapolation in recon")
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

    # Build long-form for push
    rows = []
    for t in range(len(pivot_df)):
        for nid, c in enumerate(columns):
            rows.append({"timestamp": t, "node_id": nid, "value": float(pivot_df.loc[t, c])})
    long_df = pd.DataFrame(rows)

    summary_rows = []
    cache_by_param = {}

    for v in PARAM_VALUES:
        print(f"\nParameter value = {v}  (ε for push, AoII penalty for pull),  M={M_FIXED}")

        # PUSH @ epsilon=v
        push_res = run_push_simulation(long_df, M_channels=M_FIXED, epsilon=v)
        push_met = calculate_push_metrics(push_res)
        cache_by_param[("push", v)] = (push_res, push_met)

        summary_rows.append({
            "method": "push",
            "param_value": v,
            "M": M_FIXED,
            "tx_rate": push_met["tx_rate"],
            "compression": push_met["compression"],
            "recon_rmse": push_met["recon_rmse"],
            "push_tx_rmse_z_minus_x1": push_met["transmission_rmse"],
            "attempts": push_met["attempts"],
            "successes": push_met["successes"],
            "collisions": push_met["collisions"],
            "succ_given_attempt": push_met["succ_given_attempt"],
            "coll_given_attempt": push_met["coll_given_attempt"]
        })

        # PULL @ aoii_penalty=v
        pull_res = run_pull_simulation(pivot_df, columns, M_FIXED, aoii_penalty=v)
        pull_met = calculate_pull_metrics(pull_res)
        cache_by_param[("pull", v)] = (pull_res, pull_met)

        summary_rows.append({
            "method": "pull",
            "param_value": v,
            "M": M_FIXED,
            "tx_rate": pull_met["tx_rate"],
            "compression": pull_met["compression"],
            "recon_rmse": pull_met["recon_rmse"]
        })

        print(f"  Push  -> tx_rate={push_met['tx_rate']:.6f}, recon_RMSE={push_met['recon_rmse']:.6f}, "
              f"TX_RMSE(z-x1)={push_met['transmission_rmse']:.6f}, attempts={push_met['attempts']}, "
              f"success={push_met['successes']}, collisions={push_met['collisions']}")
        print(f"  Pull  -> tx_rate={pull_met['tx_rate']:.6f}, recon_RMSE={pull_met['recon_rmse']:.6f}")

    summary = pd.DataFrame(summary_rows).sort_values(["param_value", "method"]).reset_index(drop=True)
    pd.set_option("display.float_format", lambda v: f"{v:0.6f}")
    print("\n=== Summary (M fixed at 5) ===")
    print(summary)

    # Save summary into results directory
    results_dir.mkdir(exist_ok=True)
    summary_csv_path = results_dir / "push_pull_param_sweep_summary.csv"
    summary_txt_path = results_dir / "push_pull_param_sweep_summary.txt"

    summary.to_csv(summary_csv_path, index=False)
    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("Push vs Pull param sweep summary (M=5)\n")
        f.write("=" * 80 + "\n\n")
        f.write(summary.to_string(index=False))

    print(f"\nSaved summary CSV to: {summary_csv_path}")
    print(f"Saved summary TXT to: {summary_txt_path}")

    # Sample reconstruction for param=0.5 if available
    if 0.5 in PARAM_VALUES:
        push_res, _ = cache_by_param[("push", 0.5)]
        pull_res, _ = cache_by_param[("pull", 0.5)]
        plot_sample_reconstruction(
            pivot_df,
            push_res,
            pull_res,
            node_idx=0,
            results_dir=results_dir,
            filename="sample_reconstruction_param0_5.png"
        )

    return summary


def main():
    # Assume this script is in project_root/code/
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]      # one level above /code/
    dataset_path = project_root / "data" / "synthetic_data.csv"
    results_dir = project_root / "results"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    results_dir.mkdir(exist_ok=True)

    run_param_sweep(dataset_path, results_dir)


if __name__ == "__main__":
    main()
