# -*- coding: utf-8 -*-
# Passive (exponential), Active (beta-controlled), Reward (distance-shaped)
# Replaces hardcoded kernels with your new parameterisations.
# Saves CSVs and heatmaps to ./results just like your previous version.

import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# New Parameterised Kernels (YOUR SPEC)
# -----------------------

def transition_matrix_exponential(n: int, d: float, sigma_min: float, sigma_max: float) -> np.ndarray:
    """
     - row-stochastic passive transition matrix with exponential distance decay and density-controlled width.
      - n: number of states (0..n-1)
      - d in [0,1]: dispersion / density knob (wider spread as d increases)
      - sigma_min, sigma_max: min / max decay lengths
    """
    sigma = sigma_min + d * (sigma_max - sigma_min)
    P = np.zeros((n, n), dtype=float)
    for i in range(n):
        w = np.exp(-np.abs(np.arange(n) - i) / max(sigma, 1e-12))
        P[i, :] = w / w.sum()
    return P


def get_active_transition(beta: float, n: int = 5, goal: int = 2, sigma_min: float = 0.1, sigma_max: float = 1.0) -> np.ndarray:
    """
    Active transition matrix:
      - beta=1 -> perfect control (sharp mass at goal)
      - beta=0 -> diffuse (broad around goal)
    Uses the same goal-centred exponential kernel for every row.
    """
    sigma = sigma_min + (1 - beta) * (sigma_max - sigma_min)
    P = np.zeros((n, n), dtype=float)
    w = np.exp(-np.abs(np.arange(n) - goal) / max(sigma, 1e-12))
    w = w / w.sum()
    for i in range(n):
        P[i, :] = w
    return P


import numpy as np

def generate_reward_structure(n=5, goal=2, alpha=1.0, reward_type='focused'):
    """
    Distance-shaped reward structure normalised to comparable magnitudes.

    reward_type:
      - 'focused': exponential decay (sharp)
      - 'spread' : quadratic decay (gentle)
    Scaled so both roughly align with target magnitudes (e.g., -10..0 vs -2..0).
    """
    rewards = {}
    distances = np.arange(n)
    max_d = max(goal, n-1-goal)  # furthest distance

    for s in range(n):
        d = abs(s - goal)
        if reward_type == 'focused':
            raw = -np.exp(d)
            # normalise so farthest state ≈ -10
            val = 10 * (raw / -np.exp(max_d))
        elif reward_type == 'spread':
            raw = -(d ** 2)
            # normalise so farthest state ≈ -2
            val = 2 * (raw / -max_d**2)
        else:
            raise ValueError("Unknown reward_type (use: focused|spread)")

        rewards[s] = alpha * val

    return rewards


# -----------------------
# Simulation (unchanged logic; now uses the new kernels)
# -----------------------

def sample_next_state(s, a, P_passive, P_active):
    probs = P_active[s] if a == 1 else P_passive[s]
    return np.random.choice(len(probs), p=probs)

def simulate_pull_wiql(N, M, T, P_passive, P_active, state_rewards):
    nS = P_passive.shape[0]
    states = list(range(nS))
    actions = [0, 1]
    X = [random.choice(states) for _ in range(N)]
    cumulative_reward = 0.0
    Q = [{s: {a: 0.0 for a in actions} for s in states} for _ in range(N)]
    counts = [{s: {a: 0 for a in actions} for s in states} for _ in range(N)]
    lambda_est = [{s: 0.0 for s in states} for _ in range(N)]

    for t in range(1, T + 1):
        eps = N / (N + t)                  # GLIE-style explore decay
        if random.random() < eps:
            active_arms = random.sample(range(N), min(M, N))
        else:
            priorities = [lambda_est[i][X[i]] for i in range(N)]
            active_arms = np.argsort(priorities)[-M:]

        A = [1 if i in active_arms else 0 for i in range(N)]
        step_reward = 0.0
        X_next = [None] * N

        for i in range(N):
            s, a = X[i], A[i]
            r = state_rewards[s]
            step_reward += r
            next_s = sample_next_state(s, a, P_passive, P_active)
            X_next[i] = next_s

            counts[i][s][a] += 1
            lr = 1.0 / counts[i][s][a]
            max_q_next = max(Q[i][next_s].values())
            Q[i][s][a] = (1 - lr) * Q[i][s][a] + lr * (r + max_q_next)
            lambda_est[i][s] = Q[i][s][1] - Q[i][s][0]

        cumulative_reward += step_reward /N
        X = X_next

    return cumulative_reward/T 

def simulate_push_based(N, M, T, P_passive, P_active, state_rewards):
    nS = P_passive.shape[0]
    states = list(range(nS))
    X = [random.choice(states) for _ in range(N)]
    sink_state = [None] * N
    cumulative_reward = 0.0

    for t in range(1, T + 1):
        transmit_candidates = [i for i in range(N) if sink_state[i] != X[i]]
        random.shuffle(transmit_candidates)
        active_arms = transmit_candidates[:M]
        A = [1 if i in active_arms else 0 for i in range(N)]

        X_next = [None] * N
        step_reward = 0.0

        for i in range(N):
            s, a = X[i], A[i]
            r = state_rewards[s]
            step_reward += r
            next_s = sample_next_state(s, a, P_passive, P_active)
            X_next[i] = next_s
            if A[i]:
                sink_state[i] = s

        cumulative_reward += step_reward/ N 
        X = X_next

    return cumulative_reward / T

# -----------------------
# Experiments + Heatmaps
# -----------------------

def run_experiments(nS: int, goal: int, N: int, M: int, T: int, num_runs: int,
                    reward_type: str, beta_active: float,
                    densities, alphas, sigma_passive_min: float, sigma_passive_max: float,
                    sigma_active_min: float, sigma_active_max: float) -> pd.DataFrame:
    """
    Builds passive via transition_matrix_exponential and active via get_active_transition(beta).
    """
    P_active = get_active_transition(beta=beta_active, n=nS, goal=goal,
                                     sigma_min=sigma_active_min, sigma_max=sigma_active_max)
    results = []
    for d in densities:
        P_passive = transition_matrix_exponential(n=nS, d=d,
                                                  sigma_min=sigma_passive_min, sigma_max=sigma_passive_max)
        for alpha in alphas:
            state_rewards = generate_reward_structure(n=nS, goal=goal, alpha=alpha, reward_type=reward_type)
            pull_rewards, push_rewards = [], []
            for run_id in range(num_runs):
                np.random.seed(100 + run_id)
                random.seed(100 + run_id)
                pull_final = simulate_pull_wiql(N, M, T, P_passive, P_active, state_rewards)
                push_final = simulate_push_based(N, M, T, P_passive, P_active, state_rewards)
                pull_rewards.append(pull_final)
                push_rewards.append(push_final)
            results.append({
                "density": d,
                "alpha": alpha,
                "pull_reward": float(np.mean(pull_rewards)),
                "push_reward": float(np.mean(push_rewards)),
                "difference": float(np.mean(pull_rewards) - np.mean(push_rewards)),
                "reward_type": reward_type
            })
    return pd.DataFrame(results)


def create_heatmaps(df_focused: pd.DataFrame, df_spread: pd.DataFrame, output_dir: str = "results"):
    """
    Creates six heatmaps identical in spirit to your previous version:
      - Pull (focused), Push (focused), Diff (focused),
      - Pull (spread),  Push (spread),  Diff (spread)
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set(context="notebook")
    #def _plot_one(pivot, title, cbar_label='Avg Reward', center=None, fname='heatmap.png'):
    def _plot_one(pivot, title, cmap='YlGnBu_r', cbar_label='Avg Reward', center=None, fname='heatmap.png'):
        # Keep figure size small to match final size in paper
        plt.figure(figsize=(6, 4))
        ax = plt.gca()
        
        # Set font sizes appropriate for the FINAL size in paper
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 16,
            'axes.labelsize': 16,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
        })
        
        if center is None:
            sns.heatmap(pivot, annot=False, fmt='.2f', cmap=cmap,
                        linewidths=0.5, linecolor='white',
                        cbar=True,  # Remove colorbar
                        cbar_kws={'label': cbar_label, 'shrink': 0.85},
                        annot_kws={'size': 16, 'weight': 'normal'},
                        ax=ax)
        else:
            sns.heatmap(pivot, annot=False, fmt='.2f', cmap=cmap, center=center,
                        linewidths=1.5, linecolor='white',
                        cbar=True,  # Remove colorbar
                        cbar_kws={'label': cbar_label, 'shrink': 0.85},
                        annot_kws={'size': 16, 'weight': 'normal'},
                        ax=ax)
        
        #ax.set_title(title, pad=10)
        ax.set_xlabel('Density ($d$)', labelpad=16)
        ax.set_ylabel('Reward Scale ($\\alpha$)', labelpad=16)
        ax.invert_yaxis()
        
        # Ensure tick labels are horizontal and readable
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        path = os.path.join(output_dir, fname)
        
        # Save as PDF (vector) - this is KEY for LaTeX
        pdf_path = path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf', dpi=300)
        
        # Also save PNG as backup with high DPI
        plt.savefig(path, dpi=300, bbox_inches='tight', format='png')
        
        plt.close()
        print(f"Saved: {pdf_path} (use this in LaTeX)")
        print(f"Saved: {path} (backup)")

    def _make_pivots(df):
        return (
            df.pivot(index='alpha', columns='density', values='pull_reward'),
            df.pivot(index='alpha', columns='density', values='push_reward'),
            df.pivot(index='alpha', columns='density', values='difference'),
        )

    # Focused
    if df_focused is not None and not df_focused.empty:
        piv_pull, piv_push, piv_diff = _make_pivots(df_focused)
        _plot_one(piv_pull, 'Pull-Based (Focused)', 'YlGnBu_r', 'Avg Reward', None, 'heatmap_focused_pull.png')
        _plot_one(piv_push, 'Push-Based (Focused)', 'YlGnBu_r', 'Avg Reward', None, 'heatmap_focused_push.png')
        _plot_one(piv_diff, 'Pull Advantage (Focused)', 'RdYlGn', 'Pull − Push', 0, 'heatmap_focused_diff.png')

    # Spread
    if df_spread is not None and not df_spread.empty:
        piv_pull, piv_push, piv_diff = _make_pivots(df_spread)
        _plot_one(piv_pull, 'Pull-Based (Spread)', 'YlGnBu_r', 'Avg Reward', None, 'heatmap_spread_pull.png')
        _plot_one(piv_push, 'Push-Based (Spread)', 'YlGnBu_r', 'Avg Reward', None, 'heatmap_spread_push.png')
        _plot_one(piv_diff, 'Pull Advantage (Spread)', 'RdYlGn', 'Pull − Push', 0, 'heatmap_spread_diff.png')


# -----------------------
# Main (runs locally, saves to ./results like before)
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Density × Reward-scale heatmaps with exponential kernels")
    parser.add_argument('--n_states', type=int, default=5, help='Number of states (default: 5)')
    parser.add_argument('--goal', type=int, default=2, help='Goal state index (default: 2)')
    parser.add_argument('-N', '--num_arms', type=int, default=10, help='Number of arms/nodes (default: 100)')
    parser.add_argument('-M', '--num_activations', type=int, default=2, help='Number of activations per step (default: 10)')
    parser.add_argument('-T', '--time_steps', type=int, default=5000, help='Number of time steps (default: 5000)')
    parser.add_argument('--num_runs', type=int, default=1, help='Runs to average per config (default: 10)')
    parser.add_argument('--beta_active', type=float, default=0.9, help='Active control strength β in [0,1] (default: 0.9)')
    parser.add_argument('--reward_types', type=str, default='focused,spread', help='Comma list among focused,spread,quadratic')
    parser.add_argument('--densities', type=str, default='0.1,0.3,0.6,0.9', help='Comma list of d values')
    parser.add_argument('--alphas', type=str, default='1.0,2.0,3.0,4.0,5.0', help='Comma list of α values')

    # decay parameters
    parser.add_argument('--sigma_passive_min', type=float, default=0.1, help='Passive sigma_min (default: 0.1)')
    parser.add_argument('--sigma_passive_max', type=float, default=2.0, help='Passive sigma_max (default: 2.0)')
    parser.add_argument('--sigma_active_min', type=float, default=0.1, help='Active sigma_min (default: 0.1)')
    parser.add_argument('--sigma_active_max', type=float, default=2.0, help='Active sigma_max (default: 2.0)')

    args = parser.parse_args()

    nS = args.n_states
    goal = args.goal
    densities = [float(x) for x in args.densities.split(',')]
    alphas = [float(x) for x in args.alphas.split(',')]
    reward_types = [x.strip() for x in args.reward_types.split(',')]

    print("="*60)
    print("NUMERICAL HEATMAP EXPERIMENT (Exponential Kernels): PULL vs PUSH")
    print("="*60)
    print(f"States: n={nS}, goal={goal}")
    print(f"Passive sigmas: [{args.sigma_passive_min}, {args.sigma_passive_max}]")
    print(f"Active  sigmas: [{args.sigma_active_min}, {args.sigma_active_max}]  (beta={args.beta_active})")
    print(f"Grid: {len(densities)} densities × {len(alphas)} alphas per reward type")
    print(f"Sim: N={args.num_arms}, M={args.num_activations}, T={args.time_steps}, runs={args.num_runs}")
    print("="*60)

    df_focused = None
    df_spread = None

    if 'focused' in reward_types:
        df_focused = run_experiments(nS, goal, args.num_arms, args.num_activations, args.time_steps, args.num_runs,
                                     reward_type='focused', beta_active=args.beta_active,
                                     densities=densities, alphas=alphas,
                                     sigma_passive_min=args.sigma_passive_min, sigma_passive_max=args.sigma_passive_max,
                                     sigma_active_min=args.sigma_active_min, sigma_active_max=args.sigma_active_max)

    if 'spread' in reward_types:
        df_spread = run_experiments(nS, goal, args.num_arms, args.num_activations, args.time_steps, args.num_runs,
                                    reward_type='spread', beta_active=args.beta_active,
                                    densities=densities, alphas=alphas,
                                    sigma_passive_min=args.sigma_passive_min, sigma_passive_max=args.sigma_passive_max,
                                    sigma_active_min=args.sigma_active_min, sigma_active_max=args.sigma_active_max)

    if 'quadratic' in reward_types:
        df_quad = run_experiments(nS, goal, args.num_arms, args.num_activations, args.time_steps, args.num_runs,
                                  reward_type='quadratic', beta_active=args.beta_active,
                                  densities=densities, alphas=alphas,
                                  sigma_passive_min=args.sigma_passive_min, sigma_passive_max=args.sigma_passive_max,
                                  sigma_active_min=args.sigma_active_min, sigma_active_max=args.sigma_active_max)
        # If you want to include quadratic in the heatmaps, slot it in here similarly.

    # Save results (same location/filenames as your previous version)
    os.makedirs('results', exist_ok=True)
    if df_focused is not None:
        df_focused.to_csv('results/experiment_results_focused.csv', index=False)
    if df_spread is not None:
        df_spread.to_csv('results/experiment_results_spread.csv', index=False)
    print("\nResults saved to results/experiment_results_*.csv")

    # Create heatmaps
    print("\nGenerating heatmaps...")
    create_heatmaps(df_focused if df_focused is not None else pd.DataFrame(),
                    df_spread if df_spread is not None else pd.DataFrame())

    print("\nDone.")
