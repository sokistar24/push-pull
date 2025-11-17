import os
import csv
import random
from dataclasses import dataclass
from typing import Optional, List
from collections import deque

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ==================== Config ====================

@dataclass
class Config:
    N: int      # number of nodes
    M: int      # polling budget
    K: int      # number of states (e.g. 3)
    Hcap: int   # max age bucket


# ==================== Utilities ====================

def safe_div(num, denom, default=0.0):
    try:
        if denom in (0, 0.0, None):
            return default
        return num / denom
    except ZeroDivisionError:
        return default


def compute_aoii_from_beliefs(B: np.ndarray, Y_aligned: np.ndarray):
    """
    AOII(t,n) = 0 if belief==truth at t; else AOII(t-1,n) + 1
    B, Y_aligned: shape (T', N), aligned per timestep.
    Returns:
      aoii (T', N), mean_aoii (scalar)
    """
    if B.size == 0 or Y_aligned.size == 0:
        return np.zeros((0, 0)), 0.0
    assert B.shape == Y_aligned.shape
    Tprime, N = B.shape
    aoii = np.zeros((Tprime, N), dtype=float)
    mismatch = (B != Y_aligned)
    if Tprime > 0:
        aoii[0, :] = mismatch[0, :].astype(float)
    for t in range(1, Tprime):
        aoii[t, :] = np.where(mismatch[t, :], aoii[t-1, :] + 1.0, 0.0)
    mean_aoii = float(np.mean(aoii)) if aoii.size else 0.0
    return aoii, mean_aoii


# ==================== Push-based ClassAct ====================

class PushBasedClassAct:
    """Push-based ClassAct implementation (without EWV smoothing)."""

    def __init__(self, classifier_path: str):
        self.classifier = joblib.load(classifier_path)
        self.reset()

    def reset(self):
        self.current_state = None
        self.transmitted_state = None
        self.last_transmission_time = 0
        self.transmissions = []  # list of dicts: {timestamp, transmitted_state, reason}
        self.beliefs = []        # list of dicts: {timestamp, believed_state, true_state}

    def process_sample(self, timestamp: int, accx: float, accy: float, accz: float, true_activity: int):
        accel_data = np.array([[accx, accy, accz]])
        self.current_state = self.classifier.predict(accel_data)[0]

        should_transmit = False
        reason = None
        if self.transmitted_state is None:
            should_transmit = True
            reason = "initial"
        elif self.current_state != self.transmitted_state:
            should_transmit = True
            reason = "state_change"
        elif timestamp - self.last_transmission_time > 50:
            should_transmit = True
            reason = "heartbeat"

        if should_transmit:
            self.transmissions.append({
                'timestamp': timestamp,
                'transmitted_state': int(self.current_state),
                'reason': reason
            })
            self.transmitted_state = int(self.current_state)
            self.last_transmission_time = timestamp

        self.beliefs.append({
            'timestamp': timestamp,
            'believed_state': int(self.transmitted_state) if self.transmitted_state is not None else 0,
            'true_state': int(true_activity)
        })


def run_push_classact(df_multi: pd.DataFrame, classifier_path: str, N: int):
    """
    Run a push-based ClassAct instance per node and return per-node instances + belief matrix.
    """
    T = len(df_multi)
    push_nodes = {n: PushBasedClassAct(classifier_path) for n in range(1, N + 1)}

    for t in range(1, T):
        row = df_multi.iloc[t]
        for n in range(1, N + 1):
            push_nodes[n].process_sample(
                timestamp=t,
                accx=row[f"n{n}_accx"],
                accy=row[f"n{n}_accy"],
                accz=row[f"n{n}_accz"],
                true_activity=row[f"n{n}_activity"]
            )

    belief_matrix = np.zeros((T-1, N), dtype=int)
    for n in range(1, N + 1):
        beliefs_n = [b['believed_state'] for b in push_nodes[n].beliefs]  # length T-1
        belief_matrix[:, n-1] = np.array(beliefs_n[:T-1], dtype=int)
    return push_nodes, belief_matrix


def summarise_multi_push(push_nodes: dict, belief_matrix: np.ndarray, df_multi: pd.DataFrame):
    """Metrics overall for push: missed transitions, total_messages, accuracy."""
    T = len(df_multi)
    N = belief_matrix.shape[1]

    Y = np.zeros((T, N), dtype=int)
    for n in range(1, N + 1):
        Y[:, n-1] = df_multi[f"n{n}_activity"].astype(int).to_numpy()
    Y_aligned = Y[1:1+belief_matrix.shape[0], :]

    accuracy = float(np.mean(belief_matrix == Y_aligned)) if belief_matrix.size else 0.0
    total_messages = sum(len(push_nodes[n].transmissions) for n in range(1, N + 1))

    total_transitions = 0
    detected = 0
    for n in range(1, N + 1):
        y = Y[:, n-1]
        trans_idx = np.where(y[1:] != y[:-1])[0] + 1
        total_transitions += len(trans_idx)

        # map time->transmitted states
        trans_by_time = {}
        for tr in push_nodes[n].transmissions:
            ts = tr['timestamp']
            s = tr['transmitted_state']
            trans_by_time.setdefault(ts, []).append(s)

        for k, start in enumerate(trans_idx):
            end = trans_idx[k+1] if k + 1 < len(trans_idx) else (T - 1)
            new_truth = y[start]
            detected_flag = False
            for ts in range(start, end + 1):
                if ts in trans_by_time and any(s == new_truth for s in trans_by_time[ts]):
                    detected_flag = True
                    break
            if detected_flag:
                detected += 1

    missed = max(total_transitions - detected, 0)
    return {
        'missed_transitions': int(missed),
        'total_transitions': int(total_transitions),
        'total_messages': int(total_messages),
        'accuracy': float(accuracy),
        'belief_matrix': belief_matrix,
        'truth_aligned': Y_aligned
    }


# ==================== Base Q-Learning Agent ====================

class BaseQLAgent:
    """
    Base class for per-node Q-learning agents over (s_last, delta, action).
    Q-shape: [N, K, H+1, 2]
    """

    def __init__(self, cfg: Config, gamma: float = 0.9):
        self.cfg = cfg
        self.N, self.M, self.K, self.H = cfg.N, cfg.M, cfg.K, cfg.Hcap
        self.gamma = gamma
        self.t = 0

        # Q[i, s, d, a]
        self.Q = np.zeros((self.N, self.K, self.H + 1, 2), dtype=float)
        # visit counts
        self.counts = np.zeros_like(self.Q, dtype=float)
        # λ estimates
        self.lambda_est = np.zeros((self.N, self.K, self.H + 1), dtype=float)

        self._last_obs = None  # (s_last, delta) from select()

    def _update_lambda_est(self):
        # λ(i,s,d) = Q(i,s,d,1) - Q(i,s,d,0)
        self.lambda_est = self.Q[:, :, :, 1] - self.Q[:, :, :, 0]

    def select(self, s_last: np.ndarray, delta: np.ndarray) -> List[int]:
        raise NotImplementedError

    def observe(self, s_last2: np.ndarray, delta2: np.ndarray,
                polled_idx: List[int], r_vec: np.ndarray):
        raise NotImplementedError


# ==================== Adaptive WIQL Agent (Best Pull) ====================

class WIQLAdaptiveAgent(BaseQLAgent):
    """
    WIQL-style agent:
    - epsilon(t) = max(0.05, N / (N + t))  (adaptive exploration)
    - alpha(s,d,a) = 1 / visit_count(s,d,a)
    - priorities = λ(i,s,d) = Q(i,s,d,1) - Q(i,s,d,0)  (Whittle-like index)
    """

    def __init__(self, cfg: Config, gamma: float = 0.9):
        super().__init__(cfg, gamma=gamma)

    def _eps_t(self) -> float:
        return max(0.05, float(self.N) / float(self.N + max(self.t, 1)))

    def select(self, s_last: np.ndarray, delta: np.ndarray) -> List[int]:
        self.t += 1
        self._last_obs = (s_last.copy(), delta.copy())

        eps = self._eps_t()
        # update λ estimates for all (i,s,d)
        self._update_lambda_est()

        priorities = np.zeros(self.N)
        for i in range(self.N):
            s, d = int(s_last[i]), int(delta[i])
            priorities[i] = self.lambda_est[i, s, d]

        if random.random() < eps:
            active = random.sample(range(self.N), self.M)
        else:
            active = np.argsort(priorities)[-self.M:].tolist()
        return active

    def observe(self, s_last2: np.ndarray, delta2: np.ndarray,
                polled_idx: List[int], r_vec: np.ndarray):
        s_last, delta = self._last_obs
        a = np.zeros(self.N, dtype=int)
        a[polled_idx] = 1

        for i in range(self.N):
            s, d, act = int(s_last[i]), int(delta[i]), int(a[i])
            r = float(r_vec[i])
            s2, d2 = int(s_last2[i]), int(delta2[i])

            self.counts[i, s, d, act] += 1.0
            alpha = 1.0 / self.counts[i, s, d, act]

            q = self.Q[i, s, d, act]
            td = r + self.gamma * np.max(self.Q[i, s2, d2, :])
            self.Q[i, s, d, act] = q + alpha * (td - q)

        self._update_lambda_est()


# ==================== Multi-node Pull Scheduler (Adaptive WIQL) ====================

class MultiNodeQLPullScheduler:
    """
    Generic pull-based scheduler wrapping a per-node Q-learning agent.
    It polls up to M nodes per timestep based on the agent's priorities.
    """

    def __init__(self, classifier_path: str, cfg: Config,
                 agent: BaseQLAgent, verbose: bool = False):
        self.classifier = joblib.load(classifier_path)
        self.cfg = cfg
        self.agent = agent
        self.N, self.M, self.K, self.Hcap = cfg.N, cfg.M, cfg.K, cfg.Hcap
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.t = 0
        self.belief = {n: None for n in range(1, self.N + 1)}
        self.last_query_time = {n: -1 for n in range(1, self.N + 1)}
        self.state_start_time = {n: -1 for n in range(1, self.N + 1)}

        self.decisions = []      # per timestep: {'t', 'nodes_queried', 'changes_found_nodes'}
        self.belief_matrix = []  # [(T-1) x N] accumulated beliefs after each step
        self.total_messages = 0
        self.total_samples = 0

    def _build_state_arrays(self) -> (np.ndarray, np.ndarray):
        """
        Build s_last, delta arrays for the agent from internal bookkeeping.
        s_last in [0..K-1], delta in [0..Hcap].
        """
        s_last = np.zeros(self.N, dtype=int)
        delta = np.zeros(self.N, dtype=int)
        for n in range(1, self.N + 1):
            b = self.belief[n]
            s_last[n-1] = 0 if b is None else int(np.clip(b, 0, self.K - 1))
            if self.last_query_time[n] < 0:
                d = 0
            else:
                d = self.t - self.last_query_time[n]
            delta[n-1] = int(np.clip(d, 0, self.Hcap))
        return s_last, delta

    def step(self, row: pd.Series):
        self.t += 1

        # 1) Build current state for all nodes
        s_last, delta = self._build_state_arrays()

        # 2) Ask agent which nodes to poll (indices 0..N-1)
        polled_idx = self.agent.select(s_last, delta)
        selected_nodes = [i + 1 for i in polled_idx]  # node IDs 1..N

        changed_nodes = []
        r_vec = np.zeros(self.N, dtype=float)

        # 3) Execute polls, get new beliefs and rewards
        for n in selected_nodes:
            accx = float(row[f"n{n}_accx"])
            accy = float(row[f"n{n}_accy"])
            accz = float(row[f"n{n}_accz"])
            X = np.array([[accx, accy, accz]])
            pred = int(self.classifier.predict(X)[0])

            prev = self.belief[n]
            is_first = prev is None
            state_changed = (not is_first) and (pred != prev)

            if state_changed:
                r = 1.0
                changed_nodes.append(n)
            else:
                r = -0.05  # small cost for no useful change

            r_vec[n-1] = r

            if is_first or state_changed:
                self.state_start_time[n] = self.t
            self.belief[n] = pred
            self.last_query_time[n] = self.t

        # 4) Non-polled nodes have reward 0 (already in r_vec)

        # 5) Build next state (s_last2, delta2) after this step
        s_last2, delta2 = self._build_state_arrays()

        # 6) Agent learns from transition
        self.agent.observe(s_last2, delta2, polled_idx, r_vec)

        # 7) Global counters and logs
        self.total_messages += len(selected_nodes)
        self.total_samples += self.N

        snapshot = [self.belief[n] if self.belief[n] is not None else 0
                    for n in range(1, self.N + 1)]
        self.belief_matrix.append(snapshot)

        self.decisions.append({
            't': self.t,
            'nodes_queried': selected_nodes,
            'changes_found_nodes': changed_nodes
        })

        if self.verbose and (self.t <= 10 or self.t % 500 == 0):
            print(f"t={self.t:5d}  WIQL_queried={selected_nodes}  changes={changed_nodes}")

    def run(self, df_multi: pd.DataFrame, timesteps: Optional[int] = None):
        T = len(df_multi)
        if timesteps is None:
            timesteps = T
        timesteps = min(timesteps, T)
        for i in range(1, timesteps):  # start at 1 like streaming
            self.step(df_multi.iloc[i])


# ==================== Metrics for Pull ====================

def summarise_multi_pull(sched: MultiNodeQLPullScheduler, df_multi: pd.DataFrame):
    """Metrics overall for pull: missed transitions, total_messages, accuracy."""
    N = sched.cfg.N
    T = len(df_multi)

    Y = np.zeros((T, N), dtype=int)
    for n in range(1, N + 1):
        Y[:, n-1] = df_multi[f"n{n}_activity"].astype(int).to_numpy()

    B = np.array(sched.belief_matrix, dtype=int)   # (T-1) x N
    Y_aligned = Y[1:B.shape[0]+1, :]

    accuracy = float(np.mean(B == Y_aligned)) if B.size else 0.0

    # change_found matrix
    change_found_mat = np.zeros_like(B, dtype=bool)
    for idx, dec in enumerate(sched.decisions):
        for n in dec['changes_found_nodes']:
            change_found_mat[idx, n-1] = True

    total_transitions = 0
    detected = 0
    for n in range(N):
        y = Y[:, n]
        trans_idx = np.where(y[1:] != y[:-1])[0] + 1
        total_transitions += len(trans_idx)
        for k, start in enumerate(trans_idx):
            end = trans_idx[k+1] if k + 1 < len(trans_idx) else (T - 1)
            last_row = change_found_mat.shape[0] - 1
            detected_flag = False
            for t in range(start-1, min(end, last_row + 1)):
                if change_found_mat[t, n] and B[t, n] == y[t+1]:
                    detected_flag = True
                    break
            if detected_flag:
                detected += 1

    missed = max(total_transitions - detected, 0)
    return {
        'missed_transitions': int(missed),
        'total_transitions': int(total_transitions),
        'total_messages': int(sched.total_messages),
        'accuracy': float(accuracy),
        'belief_matrix': B,
        'truth_aligned': Y_aligned
    }


# ==================== Main: Push vs Adaptive WIQL Pull (averaged runs) ====================

if __name__ == "__main__":
    data_path = "multi_node_from_single_30nodes.csv"
    classifier_path = "accelerometer_decision_tree_basic_features.pkl"

    N = 30
    M = 5
    K = 3
    Hcap = 50

    NUM_RUNS = 10  # set to 5 or 10

    df_multi = pd.read_csv(data_path)
    cfg = Config(N=N, M=M, K=K, Hcap=Hcap)
    total_samples = len(df_multi) * N

    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    pull_runs = []
    push_runs = []
    pull_aoii = []
    push_aoii = []

    for run in range(NUM_RUNS):
        print(f"Run {run+1} / {NUM_RUNS}")

        # Pull – Adaptive WIQL
        wiql_agent = WIQLAdaptiveAgent(cfg, gamma=0.9)
        pull_sched = MultiNodeQLPullScheduler(
            classifier_path=classifier_path,
            cfg=cfg,
            agent=wiql_agent,
            verbose=False
        )
        pull_sched.run(df_multi, timesteps=len(df_multi))
        pull_res = summarise_multi_pull(pull_sched, df_multi)

        B_pull = pull_res['belief_matrix']
        Y_pull = pull_res['truth_aligned']
        _, mean_aoii_pull = compute_aoii_from_beliefs(B_pull, Y_pull)

        pull_runs.append(pull_res)
        pull_aoii.append(mean_aoii_pull)

        # Push – ClassAct
        push_nodes, push_beliefs = run_push_classact(df_multi, classifier_path, N)
        push_res = summarise_multi_push(push_nodes, push_beliefs, df_multi)

        B_push = push_res['belief_matrix']
        Y_push = push_res['truth_aligned']
        _, mean_aoii_push = compute_aoii_from_beliefs(B_push, Y_push)

        push_runs.append(push_res)
        push_aoii.append(mean_aoii_push)

    # ===== Average metrics =====

    def avg(lst):
        return float(sum(lst) / len(lst)) if lst else 0.0

    metrics = {}
    for name, runs, aoii_list in [
        ("Push", push_runs, push_aoii),
        ("Pull", pull_runs, pull_aoii)
    ]:
        accs = [r['accuracy'] for r in runs]
        msgs_pct = [100.0 * safe_div(r['total_messages'], total_samples, 0.0) for r in runs]
        miss_pct = [
            100.0 * safe_div(r['missed_transitions'], max(r['total_transitions'], 1), 0.0)
            for r in runs
        ]

        metrics[name] = {
            "accuracy_pct_mean": 100.0 * avg(accs),
            "accuracy_pct_std": np.std([a*100.0 for a in accs]) if len(accs) > 1 else 0.0,
            "packets_pct_mean": avg(msgs_pct),
            "packets_pct_std": np.std(msgs_pct) if len(msgs_pct) > 1 else 0.0,
            "missed_pct_mean": avg(miss_pct),
            "missed_pct_std": np.std(miss_pct) if len(miss_pct) > 1 else 0.0,
            "aoii_mean": avg(aoii_list),
            "aoii_std": np.std(aoii_list) if len(aoii_list) > 1 else 0.0
        }

    print(f"\n=== AVERAGED METRICS over {NUM_RUNS} runs (Push vs Pull) ===")
    for name in ["Push", "Pull"]:
        m = metrics[name]
        print(f"{name:15s}: Acc={m['accuracy_pct_mean']:.2f}%  "
              f"Packets={m['packets_pct_mean']:.2f}%  "
              f"Missed={m['missed_pct_mean']:.2f}%  "
              f"AOII={m['aoii_mean']:.4f}")

    # ===== Plots (using averaged metrics) =====

    methods = ["Push", "Pull"]
    colors = ["lightblue", "orange"]
    acc_vals = [metrics[m]["accuracy_pct_mean"] for m in methods]
    pkt_vals = [metrics[m]["packets_pct_mean"] for m in methods]
    miss_vals = [metrics[m]["missed_pct_mean"] for m in methods]
    aoii_vals = [metrics[m]["aoii_mean"] for m in methods]

    # 1) Accuracy
    plt.figure(figsize=(6, 4))
    bars1 = plt.bar(range(len(methods)), acc_vals, color=colors, alpha=0.7)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.xticks(range(len(methods)), methods, fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    for i, b in enumerate(bars1):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                 f"{acc_vals[i]:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "avg_accuracy_push_vs_pull.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 2) Packets
    plt.figure(figsize=(6, 4))
    bars2 = plt.bar(range(len(methods)), pkt_vals, color=colors, alpha=0.7)
    plt.ylabel("Packets Sent (% of samples)", fontsize=14)
    plt.xticks(range(len(methods)), methods, fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    for i, b in enumerate(bars2):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                 f"{pkt_vals[i]:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "avg_packets_push_vs_pull.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 3) Missed transitions
    plt.figure(figsize=(6, 4))
    bars3 = plt.bar(range(len(methods)), miss_vals, color=colors, alpha=0.7)
    plt.ylabel("Missed transitions (%)", fontsize=14)
    plt.xticks(range(len(methods)), methods, fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    for i, b in enumerate(bars3):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                 f"{miss_vals[i]:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "avg_missed_push_vs_pull.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # 4) AOII
    plt.figure(figsize=(6, 4))
    bars4 = plt.bar(range(len(methods)), aoii_vals, color=colors, alpha=0.7)
    plt.ylabel("Mean AOII (timesteps)", fontsize=14)
    plt.xticks(range(len(methods)), methods, fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    for i, b in enumerate(bars4):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                 f"{aoii_vals[i]:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "avg_aoii_push_vs_pull.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # ===== Save per-run + averaged metrics to CSV/TXT =====

    csv_path = os.path.join(results_dir, "push_vs_pull_runs_and_mean.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Method", "Accuracy_pct", "Packets_pct", "Missed_pct", "Mean_AOII"])

        for run in range(NUM_RUNS):
            # Pull
            pr = pull_runs[run]
            acc_p = pr['accuracy'] * 100.0
            pkt_p = 100.0 * safe_div(pr['total_messages'], total_samples, 0.0)
            miss_p = 100.0 * safe_div(pr['missed_transitions'], max(pr['total_transitions'], 1), 0.0)
            writer.writerow([run+1, "Pull", acc_p, pkt_p, miss_p, pull_aoii[run]])

            # Push
            qr = push_runs[run]
            acc_q = qr['accuracy'] * 100.0
            pkt_q = 100.0 * safe_div(qr['total_messages'], total_samples, 0.0)
            miss_q = 100.0 * safe_div(qr['missed_transitions'], max(qr['total_transitions'], 1), 0.0)
            writer.writerow([run+1, "Push", acc_q, pkt_q, miss_q, push_aoii[run]])

        # Mean rows
        for name in ["Push", "Pull"]:
            m = metrics[name]
            writer.writerow([
                "MEAN", name,
                m["accuracy_pct_mean"],
                m["packets_pct_mean"],
                m["missed_pct_mean"],
                m["aoii_mean"]
            ])

    txt_path = os.path.join(results_dir, "push_vs_pull_summary.txt")
    with open(txt_path, "w") as f:
        f.write(f"=== AVERAGED METRICS over {NUM_RUNS} runs (Push vs Pull) ===\n")
        for name in ["Push", "Pull"]:
            m = metrics[name]
            f.write(
                f"{name:15s}: Acc={m['accuracy_pct_mean']:.2f}%  "
                f"Packets={m['packets_pct_mean']:.2f}%  "
                f"Missed={m['missed_pct_mean']:.2f}%  "
                f"AOII={m['aoii_mean']:.4f}\n"
            )
        f.write("\nStd deviations:\n")
        for name in ["Push", "Pull"]:
            m = metrics[name]
            f.write(
                f"{name:15s}: Acc_std={m['accuracy_pct_std']:.2f}  "
                f"Packets_std={m['packets_pct_std']:.2f}  "
                f"Missed_std={m['missed_pct_std']:.2f}  "
                f"AOII_std={m['aoii_std']:.4f}\n"
            )
