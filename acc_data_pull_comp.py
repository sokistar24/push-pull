from dataclasses import dataclass
import random
from typing import List, Tuple
import os
import csv
from typing import Optional

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ================= Config =================

@dataclass
class Config:
    N: int      # number of nodes
    M: int      # polling budget
    K: int      # number of discrete activity states (e.g. 3)
    Hcap: int   # max age bucket for delta


# ================= Utilities =================

def safe_div(num, denom, default=0.0):
    try:
        return num / denom if denom not in (0, 0.0, None) else default
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


# ================= Base Q-learning agent =================

class BaseQLAgent:
    """
    Base class for per-node Q-learning over observable state (s_last, delta).
    State space: K x (Hcap+1), actions {0=wait,1=poll}.
    Q shape: [N, K, Hcap+1, 2].
    """

    def __init__(self, cfg: Config, gamma: float = 0.9):
        self.cfg = cfg
        self.N, self.M = cfg.N, cfg.M
        self.K, self.H = cfg.K, cfg.Hcap
        self.gamma = gamma

        # Q-values and visit counts
        self.Q = np.zeros((self.N, self.K, self.H + 1, 2), dtype=float)
        self.counts = np.zeros_like(self.Q, dtype=int)

        # optional Whittle-style index estimates λ(i, s, d) = Q(s,d,1) - Q(s,d,0)
        self.lambda_est = np.zeros((self.N, self.K, self.H + 1), dtype=float)

        # slight bias towards action=1 initially
        self.Q[:, :, :, 1] = 0.2

        self._last_obs: Tuple[np.ndarray, np.ndarray] | None = None
        self.t = 0

    def select(self, s_last: np.ndarray, delta: np.ndarray) -> List[int]:
        raise NotImplementedError

    def observe(self, s_last2: np.ndarray, delta2: np.ndarray,
                polled_idx: List[int], r_vec: np.ndarray):
        raise NotImplementedError

    def _update_lambda_est(self):
        # λ(i,s,d) = Q(i,s,d,1) - Q(i,s,d,0)
        self.lambda_est = self.Q[:, :, :, 1] - self.Q[:, :, :, 0]


# ================= 1) Plain Q-learning (fixed eps, fixed alpha) =================

class PlainQLAgent(BaseQLAgent):
    """
    Plain per-node Q-learning:
    - fixed epsilon (node-level exploration)
    - fixed alpha
    - priorities = Q(s,d,1) - Q(s,d,0) at current (s,d)
    """

    def __init__(self, cfg: Config, eps: float = 0.10, alpha: float = 0.01,
                 gamma: float = 0.9):
        super().__init__(cfg, gamma=gamma)
        self.eps = eps
        self.alpha = alpha

    def select(self, s_last: np.ndarray, delta: np.ndarray) -> List[int]:
        self.t += 1
        self._last_obs = (s_last.copy(), delta.copy())

        priorities = np.zeros(self.N)
        for i in range(self.N):
            s, d = int(s_last[i]), int(delta[i])
            priorities[i] = self.Q[i, s, d, 1] - self.Q[i, s, d, 0]

        if random.random() < self.eps:
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

            q = self.Q[i, s, d, act]
            td = r + self.gamma * np.max(self.Q[i, s2, d2, :])
            self.Q[i, s, d, act] = q + self.alpha * (td - q)

        # optional λ update (not used in selection here)
        self._update_lambda_est()


# ================= 2) Adaptive WIQL-like agent (decaying eps, α = 1/visits) =================

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
        return max(0.05, self.N / (self.N + max(self.t, 1)))

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

            self.counts[i, s, d, act] += 1
            alpha = 1.0 / self.counts[i, s, d, act]

            q = self.Q[i, s, d, act]
            td = r + self.gamma * np.max(self.Q[i, s2, d2, :])
            self.Q[i, s, d, act] = q + alpha * (td - q)

        self._update_lambda_est()


# ================= 3) WIQL + UCB exploration agent =================

class WIQLUCBAgent(BaseQLAgent):
    """
    Adaptive WIQL with UCB exploration:
    - For each node i at (s,d), define:
        UCB(i,s,d,a) = Q(i,s,d,a) + c * sqrt(log(t+1) / count(i,s,d,a))
      (or c*sqrt(log(t+1)) if count=0)
    - Whittle-UCB index:  index_i = UCB(i,s,d,1) - UCB(i,s,d,0)
    - Select top-M nodes by this index.
    - alpha(s,d,a) = 1 / visit_count(s,d,a)
    """

    def __init__(self, cfg: Config, c: float = 1.0, gamma: float = 0.9):
        super().__init__(cfg, gamma=gamma)
        self.c = c

    def select(self, s_last: np.ndarray, delta: np.ndarray) -> List[int]:
        self.t += 1
        self._last_obs = (s_last.copy(), delta.copy())

        ucb_values = np.zeros(self.N)
        for i in range(self.N):
            s, d = int(s_last[i]), int(delta[i])

            ucb_action_values = np.zeros(2)
            for a in (0, 1):
                cnt = self.counts[i, s, d, a]
                if cnt > 0:
                    exploration = self.c * np.sqrt(np.log(self.t + 1.0) / cnt)
                else:
                    exploration = self.c * np.sqrt(np.log(self.t + 1.0))
                ucb_action_values[a] = self.Q[i, s, d, a] + exploration

            # Whittle-UCB index: difference between optimistic active and passive values
            ucb_values[i] = ucb_action_values[1] - ucb_action_values[0]

        active = np.argsort(ucb_values)[-self.M:].tolist()
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

            self.counts[i, s, d, act] += 1
            alpha = 1.0 / self.counts[i, s, d, act]

            q = self.Q[i, s, d, act]
            td = r + self.gamma * np.max(self.Q[i, s2, d2, :])
            self.Q[i, s, d, act] = q + alpha * (td - q)

        self._update_lambda_est()


# ================= Multi-node Q Pull Scheduler =================

class MultiNodeQLPullScheduler:
    """
    Pull-based scheduler using a Q-learning agent (Plain, WIQL-Adaptive, or WIQL-UCB).
    Matches the interface needed by summarise_multi_pull:
    - N, total_messages
    - belief_matrix (T-1 x N)
    - decisions: list of {'t', 'nodes_queried', 'changes_found_nodes'}
    """

    def __init__(self, classifier_path: str, cfg: Config, agent: BaseQLAgent,
                 verbose: bool = False):
        self.classifier = joblib.load(classifier_path)
        self.cfg = cfg
        self.N, self.M = cfg.N, cfg.M
        self.K, self.Hcap = cfg.K, cfg.Hcap
        self.agent = agent
        self.verbose = verbose
        self.reset()

    def reset(self):
        self.t = 0
        self.belief = {n: None for n in range(1, self.N + 1)}
        self.last_query_time = {n: -1 for n in range(1, self.N + 1)}
        self.state_start_time = {n: -1 for n in range(1, self.N + 1)}

        self.decisions = []
        self.belief_matrix = []
        self.total_messages = 0
        self.total_samples = 0

    def _build_state_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build s_last, delta arrays for the agent from internal bookkeeping.
        s_last in [0..K-1], delta in [0..Hcap].
        """
        s_last = np.zeros(self.N, dtype=int)
        delta = np.zeros(self.N, dtype=int)
        for n in range(1, self.N + 1):
            b = self.belief[n]
            s_last[n-1] = 0 if b is None else int(np.clip(b, 0, self.K-1))

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

        # 2) Ask agent which nodes to poll (0-based indices)
        polled_idx = self.agent.select(s_last, delta)
        selected_nodes = [i + 1 for i in polled_idx]

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
                r = 2.0
                changed_nodes.append(n)
            else:
                r = -0.5  # small cost for a non-informative poll

            r_vec[n-1] = r

            if is_first or state_changed:
                self.state_start_time[n] = self.t
            self.belief[n] = pred
            self.last_query_time[n] = self.t

        # 4) Not polled nodes keep default reward = 0.0 in r_vec

        # 5) Build next state after this step
        s_last2, delta2 = self._build_state_arrays()

        # 6) Let agent learn from transition
        self.agent.observe(s_last2, delta2, polled_idx, r_vec)

        # 7) Update metrics
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
            print(f"t={self.t:5d}  queried={selected_nodes}  changes={changed_nodes}")

    def run(self, df_multi: pd.DataFrame, timesteps: Optional[int] = None):
        T = len(df_multi)
        if timesteps is None:
            timesteps = T
        timesteps = min(timesteps, T)
        for i in range(1, timesteps):  # start at 1 like streaming
            self.step(df_multi.iloc[i])


# ================= Metrics for pull =================

def summarise_multi_pull(sched: MultiNodeQLPullScheduler, df_multi: pd.DataFrame):
    """Metrics overall for pull: missed transitions, total_messages, accuracy."""
    N = sched.N
    T = len(df_multi)

    Y = np.zeros((T, N), dtype=int)
    for n in range(1, N + 1):
        Y[:, n-1] = df_multi[f"n{n}_activity"].astype(int).to_numpy()

    B = np.array(sched.belief_matrix, dtype=int)   # (T-1) x N
    Y_aligned = Y[1:B.shape[0]+1, :]

    accuracy = float(np.mean(B == Y_aligned)) if B.size else 0.0

    # mark which timesteps had a correctly detected change for each node
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


# ================= Main: compare three pull strategies =================

if __name__ == "__main__":
    data_path = "multi_node_from_single_30nodes.csv"
    classifier_path = "accelerometer_decision_tree_basic_features.pkl"

    N = 30
    M = 3
    K = 3
    Hcap = 50

    df_multi = pd.read_csv(data_path)
    cfg = Config(N=N, M=M, K=K, Hcap=Hcap)

    # 1) Plain Q-learning pull
    plain_agent = PlainQLAgent(cfg, eps=0.10, alpha=0.01, gamma=0.9)
    plain_sched = MultiNodeQLPullScheduler(classifier_path, cfg, agent=plain_agent, verbose=False)
    plain_sched.run(df_multi, timesteps=len(df_multi))
    plain_results = summarise_multi_pull(plain_sched, df_multi)

    # 2) Adaptive WIQL-style pull
    wiql_adapt_agent = WIQLAdaptiveAgent(cfg, gamma=0.9)
    wiql_adapt_sched = MultiNodeQLPullScheduler(classifier_path, cfg, agent=wiql_adapt_agent, verbose=False)
    wiql_adapt_sched.run(df_multi, timesteps=len(df_multi))
    wiql_adapt_results = summarise_multi_pull(wiql_adapt_sched, df_multi)

    # 3) WIQL + UCB exploration pull
    wiql_ucb_agent = WIQLUCBAgent(cfg, c=2.0, gamma=0.9)
    wiql_ucb_sched = MultiNodeQLPullScheduler(classifier_path, cfg, agent=wiql_ucb_agent, verbose=False)
    wiql_ucb_sched.run(df_multi, timesteps=len(df_multi))
    wiql_ucb_results = summarise_multi_pull(wiql_ucb_sched, df_multi)

    # Collect headline metrics
    results = {
        'PlainQ':     {k: plain_results[k]      for k in ('missed_transitions','total_transitions','total_messages','accuracy')},
        'WIQL-Adapt': {k: wiql_adapt_results[k] for k in ('missed_transitions','total_transitions','total_messages','accuracy')},
        'WIQL-UCB':   {k: wiql_ucb_results[k]   for k in ('missed_transitions','total_transitions','total_messages','accuracy')}
    }

    # AOII for each
    B_plain, Y_plain = plain_results['belief_matrix'], plain_results['truth_aligned']
    _, mean_aoii_plain = compute_aoii_from_beliefs(B_plain, Y_plain)

    B_adapt, Y_adapt = wiql_adapt_results['belief_matrix'], wiql_adapt_results['truth_aligned']
    _, mean_aoii_adapt = compute_aoii_from_beliefs(B_adapt, Y_adapt)

    B_ucb, Y_ucb = wiql_ucb_results['belief_matrix'], wiql_ucb_results['truth_aligned']
    _, mean_aoii_ucb = compute_aoii_from_beliefs(B_ucb, Y_ucb)

    # ============== Metrics precomputation ==============

    methods = list(results.keys())
    colors = ['orange', 'green', 'purple'][:len(methods)]
    total_samples = len(df_multi) * N

    accuracies = [results[m]['accuracy'] * 100 for m in methods]

    message_percentages = [
        (results[m]['total_messages'] / max(total_samples, 1)) * 100
        for m in methods
    ]

    miss_rates = []
    for m in methods:
        tot_tr = max(results[m].get('total_transitions', 0), 1)
        miss = results[m]['missed_transitions'] / tot_tr * 100.0
        miss_rates.append(miss)

    aoii_values = [mean_aoii_plain, mean_aoii_adapt, mean_aoii_ucb]
    labels = ['PlainQ','WIQL-Adapt','WIQL-UCB']
    aoii_map = {
        'PlainQ': mean_aoii_plain,
        'WIQL-Adapt': mean_aoii_adapt,
        'WIQL-UCB': mean_aoii_ucb
    }

    # ============== Create Results folder ==============

    results_dir = "Results"
    os.makedirs(results_dir, exist_ok=True)

    # ============== Plots: 4 metrics (saved to Results) ==============

    # 1) Accuracy (%)
    plt.figure(figsize=(6, 4))
    bars1 = plt.bar(range(len(methods)), accuracies, color=colors, alpha=0.7)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.ylim([0.0, max(115, (max(accuracies) + 5 if accuracies else 100))])
    plt.xticks(range(len(methods)), methods, fontsize=12, rotation=0)
    plt.grid(axis='y', alpha=0.3)
    for i, b in enumerate(bars1):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                 f'{accuracies[i]:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2) Packets / messages sent (% of total samples across all nodes)
    plt.figure(figsize=(6, 4))
    max_percentage = max(message_percentages) if message_percentages else 1
    bars2 = plt.bar(range(len(methods)), message_percentages, color=colors, alpha=0.7)
    plt.ylabel('Packets Sent (% of samples)', fontsize=14)
    plt.ylim([0, max_percentage * 1.15])
    plt.xticks(range(len(methods)), methods, fontsize=12, rotation=0)
    plt.grid(axis='y', alpha=0.3)
    for i, b in enumerate(bars2):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                 f'{message_percentages[i]:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "packets.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3) Missed detection rate (% of true transitions)
    plt.figure(figsize=(6, 4))
    max_rate = max(miss_rates) if miss_rates else 1
    bars3 = plt.bar(range(len(methods)), miss_rates, color=colors, alpha=0.7)
    plt.ylabel('Missed transitions (%)', fontsize=14)
    plt.ylim([0, max_rate * 1.15])
    plt.xticks(range(len(methods)), methods, fontsize=12, rotation=0)
    plt.grid(axis='y', alpha=0.3)
    for i, b in enumerate(bars3):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                 f'{miss_rates[i]:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "missed_transitions.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4) AOII (mean across all node-time pairs; lower is better)
    plt.figure(figsize=(6, 4))
    ymax = max(aoii_values) if aoii_values else 1
    bars4 = plt.bar(range(len(labels)), aoii_values, color=colors, alpha=0.7)
    plt.ylabel('Mean AOII (timesteps)', fontsize=14)
    plt.ylim([0, ymax * 1.15 if ymax > 0 else 1])
    plt.xticks(range(len(labels)), labels, fontsize=12, rotation=0)
    plt.grid(axis='y', alpha=0.3)
    for i, b in enumerate(bars4):
        plt.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                 f'{aoii_values[i]:.2f}', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "aoii.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ============== Print summary to console ==============

    print("\n=== METRICS (Three Pull Policies: PlainQ vs WIQL-Adapt vs WIQL-UCB) ===")
    for name, r in results.items():
        msg_pct = 100 * safe_div(r['total_messages'], total_samples, 0.0)
        miss_rate = 100 * safe_div(r['missed_transitions'], max(r['total_transitions'], 1), 0.0)
        print(f"{name:>11}:  Acc={r['accuracy']*100:.1f}%  "
              f"Packets={msg_pct:.1f}%  Missed={miss_rate:.1f}%")

    print(f"\nMean AOII  PlainQ    : {mean_aoii_plain:.4f}")
    print(f"Mean AOII  WIQL-Adapt: {mean_aoii_adapt:.4f}")
    print(f"Mean AOII  WIQL-UCB  : {mean_aoii_ucb:.4f}")

    # ============== Save metrics to CSV and TXT in Results ==============

    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Accuracy_pct", "Packets_pct", "Missed_pct", "Mean_AOII"])
        for name in methods:
            r = results[name]
            acc_pct = r['accuracy'] * 100.0
            pkt_pct = 100 * safe_div(r['total_messages'], total_samples, 0.0)
            miss_pct = 100 * safe_div(r['missed_transitions'], max(r['total_transitions'], 1), 0.0)
            writer.writerow([name, acc_pct, pkt_pct, miss_pct, aoii_map[name]])

    txt_path = os.path.join(results_dir, "metrics_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=== METRICS (Three Pull Policies: PlainQ vs WIQL-Adapt vs WIQL-UCB) ===\n")
        for name, r in results.items():
            msg_pct = 100 * safe_div(r['total_messages'], total_samples, 0.0)
            miss_rate = 100 * safe_div(r['missed_transitions'], max(r['total_transitions'], 1), 0.0)
            f.write(f"{name:>11}:  Acc={r['accuracy']*100:.1f}%  "
                    f"Packets={msg_pct:.1f}%  Missed={miss_rate:.1f}%\n")
        f.write("\nMean AOII:\n")
        f.write(f"PlainQ     : {mean_aoii_plain:.4f}\n")
        f.write(f"WIQL-Adapt : {mean_aoii_adapt:.4f}\n")
        f.write(f"WIQL-UCB   : {mean_aoii_ucb:.4f}\n")
