# compare_push_vs_wiql_heatmaps.py
# Push (event-triggered) vs Pull (WIQL) — average-reward heatmaps over density d and penalty λ.

import argparse, itertools, math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# -----------------------------
# Core configuration
# -----------------------------
@dataclass
class Config:
    # problem size
    N: int = 50
    M: int = 5
    K: int = 5
    # dwell times per state (similar magnitude -> consistent time-scales)
    mu: Tuple[float, ...] = (6.0, 9.0, 12.0, 16.0, 20.0)
    # value model
    v0: float = 2.0
    half_life_factor: float = 1.0  # half_life = half_life_factor * mean(mu)
    # AoI cap
    Hcap_mult: float = 5.0         # Hcap = ceil(Hcap_mult * max(mu))
    # horizon
    T: int = 2000
    # RNG seed
    seed: int = 123

    # derived (filled later)
    half_life: float = 0.0
    Hcap: int = 0

    def derive(self):
        mu_arr = np.array(self.mu, dtype=float)
        self.half_life = float(self.half_life_factor * mu_arr.mean())
        self.Hcap = int(math.ceil(self.Hcap_mult * float(mu_arr.max())))

# -----------------------------
# Environment
# -----------------------------
class Env:
    """
    K-state ring. In each step, state i stays with prob p_stay[s]=1-1/mu[s]; else jumps by ±1 with prob d
    or ±2 with prob 1-d (sign equiprobable).
    Observables at the sink:
      - s_last[i]: last delivered (push) or last polled (pull) state
      - delta[i]: steps since last delivery/poll, capped at Hcap
    Hidden:
      - tau[i]: steps since true state diverged from s_last[i] while not delivered/polled
    """
    def __init__(self, cfg: Config, d: float, lam: float):
        self.cfg = cfg
        self.d = float(d)
        self.lam = float(lam)
        self.rng = np.random.RandomState(cfg.seed)
        self.K = cfg.K
        self.mu = np.array(cfg.mu, dtype=float)
        self.p_stay = 1.0 - 1.0 / self.mu
        self.reset()

    def reset(self):
        N = self.cfg.N
        self.s = self.rng.randint(0, self.K, size=N)      # true hidden state
        self.s_last = self.s.copy()                        # last known at sink
        self.delta = np.zeros(N, dtype=int)                # AoI since last delivery/poll
        self.tau: List[Optional[int]] = [None] * self.cfg.N  # change-age since divergence
        self.total_reward = 0.0
        self.t = 0

    def _jump_offset(self) -> int:
        if self.rng.rand() < self.d:
            return 1 if self.rng.rand() < 0.5 else -1
        else:
            return 2 if self.rng.rand() < 0.5 else -2

    def _evolve_one(self, i: int):
        s = self.s[i]
        if self.rng.rand() >= self.p_stay[s]:
            self.s[i] = (s + self._jump_offset()) % self.K

    @staticmethod
    def value_of_change(v0: float, half_life: float, tau: int) -> float:
        if tau <= 0:
            return v0
        return v0 * (0.5 ** (tau / max(half_life, 1e-9)))

    def step_push(self) -> float:
        """
        Push/events: any node whose true state != s_last is a candidate.
        Capacity M: pick the M with largest tau to maximise immediate value.
        Reward = sum(value(tau) - lam) over transmitted nodes.
        """
        N, H, lam = self.cfg.N, self.cfg.Hcap, self.lam
        # evolve true dynamics
        for i in range(N):
            self._evolve_one(i)
        # update hidden change-ages
        for i in range(N):
            if self.s[i] != self.s_last[i]:
                self.tau[i] = 1 if self.tau[i] is None else self.tau[i] + 1
        # choose up to M candidates by largest tau
        cand = [(self.tau[i] if self.tau[i] is not None else -1, i) for i in range(N)]
        cand = [(t, i) for (t, i) in cand if t >= 1]
        cand.sort(reverse=True)
        chosen = [i for (_, i) in cand[: self.cfg.M]]

        step_rew = 0.0
        delivered = np.zeros(N, dtype=bool)
        delivered[chosen] = True

        # compute rewards and update sink knowledge
        for i in chosen:
            val = self.value_of_change(self.cfg.v0, self.cfg.half_life, self.tau[i])
            step_rew += (val - lam)
            self.s_last[i] = self.s[i]
            self.delta[i] = 0
            self.tau[i] = None

        # AoI increments for non-delivered
        self.delta[~delivered] = np.minimum(self.delta[~delivered] + 1, H)

        self.total_reward += step_rew
        self.t += 1
        return step_rew

    # ---------------- pull / WIQL support ----------------
    def snapshot_pull(self):
        # state given to the learner (observable)
        return self.s_last.copy(), self.delta.copy()

    def reveal_and_reward(self, chosen: List[int]) -> Tuple[np.ndarray, float]:
        """
        Execute a pull action: poll 'chosen' indices.
        Reward per polled i: value(tau[i]) - lam if there was a change, else -lam.
        Returns (per-node reward vector, step total)
        """
        N, H, lam = self.cfg.N, self.cfg.Hcap, self.lam
        # evolve underlying truth first
        for i in range(N):
            self._evolve_one(i)
        # update change-ages
        for i in range(N):
            if self.s[i] != self.s_last[i]:
                self.tau[i] = 1 if self.tau[i] is None else self.tau[i] + 1

        r = np.zeros(N, dtype=float)
        polled = np.zeros(N, dtype=bool)
        polled[chosen] = True
        # rewards and reveal
        for i in chosen:
            if self.tau[i] is None:
                r[i] = -lam
            else:
                val = self.value_of_change(self.cfg.v0, self.cfg.half_life, self.tau[i])
                r[i] = val - lam
            self.s_last[i] = self.s[i]
            self.delta[i] = 0
            self.tau[i] = None

        # AoI increments for non-polled
        self.delta[~polled] = np.minimum(self.delta[~polled] + 1, H)

        step_rew = float(r.sum())
        self.total_reward += step_rew
        self.t += 1
        return r, step_rew

# -----------------------------
# WIQL policy (pull)
# -----------------------------
class WIQL:
    """
    Per-node Q-learning on observable state (s_last, delta), actions {0=wait,1=poll}.
    Priority = Q(s,d,1) - Q(s,d,0). Choose top-M each step.
    """
    def __init__(self, cfg: Config, eps=0.10, alpha=0.25, gamma=0.95):
        self.cfg = cfg
        self.eps, self.alpha, self.gamma = eps, alpha, gamma
        self.K, self.H = cfg.K, cfg.Hcap
        self.Q = [np.zeros((self.K, self.H + 1, 2), dtype=float) for _ in range(cfg.N)]
        for i in range(cfg.N):
            self.Q[i][:, :, 1] = 0.2  # nudge towards exploring action=1
        self._last_obs = None
        self._last_act = None

    def select(self, s_last: np.ndarray, delta: np.ndarray) -> List[int]:
        self._last_obs = (s_last.copy(), delta.copy())
        # epsilon sampling over nodes for some exploration of polls
        if random.random() < self.eps:
            return random.sample(range(self.cfg.N), self.cfg.M)
        pr = [self.Q[i][s_last[i], delta[i], 1] - self.Q[i][s_last[i], delta[i], 0]
              for i in range(self.cfg.N)]
        return np.argsort(pr)[-self.cfg.M:].tolist()

    def observe(self, s_last2: np.ndarray, delta2: np.ndarray, polled: List[int], r_vec: np.ndarray):
        s_last, delta = self._last_obs
        a = np.zeros(self.cfg.N, dtype=int); a[polled] = 1
        for i in range(self.cfg.N):
            s, d, act = s_last[i], delta[i], a[i]
            r = float(r_vec[i])
            s2, d2 = s_last2[i], delta2[i]
            q = self.Q[i][s, d, act]
            td = r + self.gamma * np.max(self.Q[i][s2, d2])
            self.Q[i][s, d, act] = q + self.alpha * (td - q)

# -----------------------------
# Experiment loops
# -----------------------------
def run_push(cfg: Config, d: float, lam: float, runs: int, T: int) -> float:
    scores = []
    for rep in range(runs):
        local = Config(N=cfg.N, M=cfg.M, K=cfg.K, mu=cfg.mu, v0=cfg.v0,
                       half_life_factor=cfg.half_life_factor, Hcap_mult=cfg.Hcap_mult,
                       T=T, seed=cfg.seed + 1000 + rep)
        local.derive()
        env = Env(local, d=d, lam=lam)
        total = 0.0
        for _ in range(T):
            total += env.step_push()
        scores.append(total / T)
    return float(np.mean(scores))

def run_pull_wiql(cfg: Config, d: float, lam: float, runs: int, T: int, warmup: int = 200) -> float:
    scores = []
    for rep in range(runs):
        local = Config(N=cfg.N, M=cfg.M, K=cfg.K, mu=cfg.mu, v0=cfg.v0,
                       half_life_factor=cfg.half_life_factor, Hcap_mult=cfg.Hcap_mult,
                       T=T, seed=cfg.seed + 2000 + rep)
        local.derive()
        env = Env(local, d=d, lam=lam)
        agent = WIQL(local, eps=0.10, alpha=0.25, gamma=0.95)

        total = 0.0
        for t in range(T):
            s_last, delta = env.snapshot_pull()
            chosen = agent.select(s_last, delta)
            r_vec, step = env.reveal_and_reward(chosen)
            s_last2, delta2 = env.snapshot_pull()
            agent.observe(s_last2, delta2, chosen, r_vec)
            if t >= warmup:
                total += step
        eff_T = max(1, T - warmup)
        scores.append(total / eff_T)
    return float(np.mean(scores))

def heatmap(d_list, lam_list, vals, title: str, fname: str):
    import numpy as np
    import matplotlib.pyplot as plt
    if plt is None:
        print("matplotlib not available; skipping plot.")
        return

    # Ensure axes are sorted and match vals shape (rows=y, cols=x)
    d_vals   = np.array(sorted(d_list))                  # x-axis (columns)
    lam_vals = np.array(sorted(lam_list))                # y-axis (rows)
    Z = np.asarray(vals)
    assert Z.shape == (lam_vals.size, d_vals.size), "vals must be (len(lam_list), len(d_list))"

    # Build boundaries so pcolormesh draws full cells with edges
    def _to_edges(v):
        v = np.asarray(v, dtype=float)
        dv = np.diff(v)
        left  = v[0] - dv[0]/2
        right = v[-1] + dv[-1]/2
        mids  = v[:-1] + dv/2
        return np.concatenate([[left], mids, [right]])

    x_edges = _to_edges(d_vals)
    y_edges = _to_edges(lam_vals)

    plt.figure(figsize=(6, 4))
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
    })

    # Draw cells with white borders for the grid effect
    pcm = plt.pcolormesh(
        x_edges, y_edges, Z,
        shading='flat',              # one colour per cell
        edgecolors='white',          # grid lines
        linewidth=0.5
    )

    cbar = plt.colorbar(pcm, label='Ave Reward', shrink=0.85)
    cbar.ax.tick_params(labelsize=14)

    plt.xlabel('Density ($d$)', labelpad=10)
    plt.ylabel('Penalty ($\\lambda$)', labelpad=10)

    # Put ticks exactly at your parameter values
    ax = plt.gca()
    ax.set_xticks(d_vals)
    ax.set_yticks(lam_vals)

    # Optional: rotate x tick labels if they overlap
    # plt.setp(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()

    pdf_fname = fname.replace('.png', '.pdf')
    plt.savefig(pdf_fname, bbox_inches='tight', format='pdf', dpi=300)
    plt.savefig(fname, bbox_inches='tight', format='png', dpi=300)
    plt.close()
    print(f"Saved {pdf_fname}\nSaved {fname}")


# -----------------------------
# CLI and main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Density × penalty heatmaps: Push vs Pull (WIQL)")
    parser.add_argument('--n_states', type=int, default=5, help='Number of states (default: 5)')
    parser.add_argument('-N', '--num_arms', type=int, default=50, help='Number of nodes (default: 50)')
    parser.add_argument('-M', '--num_activations', type=int, default=5, help='Capacity per step (default: 5)')
    parser.add_argument('-T', '--time_steps', type=int, default=2000, help='Time steps per run (default: 2000)')
    parser.add_argument('--num_runs', type=int, default=3, help='Runs to average (default: 3)')
    parser.add_argument('--densities', type=str, default='0.1,0.3,0.6,0.9',
                        help='Comma list of d values')
    parser.add_argument('--lambdas', type=str, default='0.1,0.30,0.50,0.70,0.90',
                        help='Comma list of λ values (absolute, same units as reward)')
    parser.add_argument('--half_life_factors', type=str, default='1.0',
                        help='Comma list of half-life factors; we plot the first, loop others in console')
    parser.add_argument('--seed', type=int, default=123, help='Base RNG seed')
    args = parser.parse_args()

    densities = [float(x) for x in args.densities.split(',')]
    lam_list = [float(x) for x in args.lambdas.split(',')]
    hl_factors = [float(x) for x in args.half_life_factors.split(',')]

    print("="*60)
    print("HEATMAP EXPERIMENT: PUSH (event) vs PULL (WIQL)")
    print("="*60)
    print(f"K={args.n_states}, N={args.num_arms}, M={args.num_activations}, T={args.time_steps}, runs={args.num_runs}")
    print(f"d values: {densities}")
    print(f"λ values: {lam_list}")
    print(f"half-life factors: {hl_factors}")
    print("="*60)

    # base config
    base = Config(N=args.num_arms, M=args.num_activations, K=args.n_states,
                  mu=(6.0, 9.0, 12.0, 16.0, 20.0)[:args.n_states],
                  v0=2.0, half_life_factor=hl_factors[0], Hcap_mult=5.0,
                  T=args.time_steps, seed=args.seed)
    base.derive()

    # compute heatmaps for the first half-life factor; loop others in console (no extra plots to keep it simple)
    for idx, hlf in enumerate(hl_factors):
        base.half_life_factor = hlf
        base.derive()
        print(f"\n--- half-life factor = {hlf:.2f} (half_life ≈ {base.half_life:.2f}) ---")

        push_mat = np.zeros((len(lam_list), len(densities)), dtype=float)
        pull_mat = np.zeros_like(push_mat)

        for iL, lam in enumerate(lam_list):
            for jD, d in enumerate(densities):
                push_avg = run_push(base, d=d, lam=lam, runs=args.num_runs, T=args.time_steps)
                pull_avg = run_pull_wiql(base, d=d, lam=lam, runs=args.num_runs, T=args.time_steps)
                push_mat[iL, jD] = push_avg
                pull_mat[iL, jD] = pull_avg
                print(f"d={d:.2f} λ={lam:.2f}  push={push_avg:.4f}  pull={pull_avg:.4f}")

        # plots for the first factor only
        tag = f"hl{hlf:.2f}".replace('.', '')
        if idx == 0:
            heatmap(densities, lam_list, push_mat, f"Push (event) — avg reward [{tag}]", f"heatmap_push_{tag}.png")
            heatmap(densities, lam_list, pull_mat, f"Pull (WIQL) — avg reward [{tag}]", f"heatmap_pull_{tag}.png")
            diff = pull_mat - push_mat
            heatmap(densities, lam_list, diff, f"Pull − Push (avg reward) [{tag}]", f"heatmap_diff_{tag}.png")

if __name__ == "__main__":
    main()
