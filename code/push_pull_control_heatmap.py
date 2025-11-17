import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from typing import Optional

# -----------------------
# Environment Setup
# -----------------------
states = [0, 1, 2, 3, 4]
actions = [0, 1]

# -----------------------
# Parameterized Configuration Functions
# -----------------------




def generate_passive_transitions_smooth(
    density: float,                # now expected in [0,1]
    n: int = 5,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
    p: float = 1.0,                # 1=exp, 2≈Gaussian tails
    cap_hops: Optional[int] = 2,   # None = full support; 2 mimics ±2 jumps
    ring: bool = False             # True = circular (wrap-around) distance
) -> np.ndarray:
    d = float(np.clip(density, 0.0, 1.0))
    sigma = max(sigma_min + d * (sigma_max - sigma_min), 1e-12)

    P = np.zeros((n, n), dtype=float)
    idx = np.arange(n)

    for i in range(n):
        if ring:
            dist = np.minimum(np.abs(idx - i), n - np.abs(idx - i))
        else:
            dist = np.abs(idx - i)

        w = np.exp(- (dist / sigma) ** p)

        if cap_hops is not None:
            w[dist > cap_hops] = 0.0

        s = w.sum()
        P[i, :] = w / s if s > 0 else np.eye(n)[i]

    return P


def generate_passive_transitions_exp(
    density: float,
    n: int = 5,
    sigma_min: float = 0.0,
    sigma_max: float = 1.0,
    cap_hops: Optional[int] = 2,     # set to None for full support
    ring: bool = False            # use circular distance if True
) -> np.ndarray:
    """
    Exponential-distance passive transition matrix (row-stochastic),
    controlled by a single 'density' knob mapped to sigma.

    density in [0.1, 0.9]: higher -> broader spread
    sigma(density) = sigma_min + d * (sigma_max - sigma_min),
    with d = (density - 0.1) / (0.9 - 0.1) clipped to [0,1].

    cap_hops: if not None, zero out probabilities for |Δ| > cap_hops
              (mimics original ±2 support).
    ring:     if True, use circular distance on a ring; else linear chain.
    """
    # map density -> d ∈ [0,1] -> sigma
    d = (density - 0.1) / 0.8
    d = float(np.clip(d, 0.0, 1.0))
    sigma = sigma_min + d * (sigma_max - sigma_min)
    sigma = max(sigma, 1e-12)

    P = np.zeros((n, n), dtype=float)

    for i in range(n):
        idx = np.arange(n)
        if ring:
            dist = np.minimum(np.abs(idx - i), n - np.abs(idx - i))
        else:
            dist = np.abs(idx - i)

        # exponential weights
        w = np.exp(-dist / sigma)

        # optionally cap to ±cap_hops
        if cap_hops is not None:
            w[dist > cap_hops] = 0.0

        # normalise row
        s = w.sum()
        if s == 0.0:
            # fallback: stay-put if everything got zeroed (shouldn't happen with cap_hops>=0)
            P[i, i] = 1.0
        else:
            P[i, :] = w / s

    return P


import numpy as np

def generate_reward_structure(alpha, reward_type='focused', n=5, goal=2):
    rewards = {}
    for s in range(n):
        d = abs(s - goal)

        if reward_type == 'focused':
            # Matches paper: f(d) = exp(d)
            val = -alpha * np.exp(d)
        elif reward_type == 'spread':
            # Matches paper: f(d) = d
            val = -alpha * d
        else:
            raise ValueError("reward_type must be 'focused' or 'spread'")

        # Zero reward at goal
        if d == 0:
            val = 0.0

        rewards[s] = val

    return rewards



def get_active_transition(strength='perfect'):
    """Define active action transition matrix"""
    if strength == 'perfect':
        P_active = np.zeros((5, 5))
        P_active[:, 2] = 1.0  # Always go to goal
    elif strength == 'strong':
        P_active = np.array([
            [0.1, 0.0, 0.90, 0.00, 0.00],
            [0.00, 0.10, 0.90, 0.00, 0.00],
            [0.00, 0.00, 1.0, 0.00, 0.00],
            [0.00, 0.00, 0.9, 0.10, 0.00],
            [0.00, 0.00, 0.90, 0.0, 0.1]
        ])
    else:  # 'moderate'
        P_active = np.array([
            [0.10, 0.30, 0.50, 0.08, 0.02],
            [0.05, 0.25, 0.60, 0.08, 0.02],
            [0.02, 0.15, 0.66, 0.15, 0.02],
            [0.02, 0.08, 0.60, 0.25, 0.05],
            [0.02, 0.08, 0.50, 0.30, 0.10]
        ])
    
    return P_active

import numpy as np
from typing import Optional

def generate_active_transitions(
    beta: float,                 # control parameter in [0,1]
    n: int = 5,
    goal: int = 2,
    sigma_min: float = 0.1,
    sigma_max: float = 2.0,
    p: float = 1.0,              # 1 = exp, 2 = Gaussian-like
    ring: bool = False
) -> np.ndarray:
    """
    Generate active transition matrix P1(s'|s; β) based on proximity to goal state.
    
    Each row is identical: transitions favour the goal state s_g.
    """
    beta = float(np.clip(beta, 0.0, 1.0))
    sigma = max(sigma_min + beta * (sigma_max - sigma_min), 1e-12)

    idx = np.arange(n)

    # Distance measured from the goal (not from current state)
    if ring:
        dist_goal = np.minimum(np.abs(idx - goal), n - np.abs(idx - goal))
    else:
        dist_goal = np.abs(idx - goal)

    # Weight for each possible next state s'
    w = np.exp(- (dist_goal / sigma) ** p)
    w /= w.sum()

    # All rows are identical since all states share the same goal-directed dynamics
    P = np.tile(w, (n, 1))

    return P

# -----------------------

def sample_next_state(s, a, P_passive, P_active):
    """Sample next state given current state and action"""
    probs = P_active[s] if a == 1 else P_passive[s]
    return np.random.choice(states, p=probs)

def simulate_pull_wiql(N, M, T, P_passive, P_active, state_rewards):
    """Pull-based WIQL simulation"""
    X = [random.choice(states) for _ in range(N)]
    cumulative_reward = 0.0

    Q = [{s: {a: 0.0 for a in actions} for s in states} for _ in range(N)]
    counts = [{s: {a: 0 for a in actions} for s in states} for _ in range(N)]
    lambda_est = [{s: 0.0 for s in states} for _ in range(N)]

    for t in range(1, T + 1):
        eps = N / (N + t)
        if random.random() < eps:
            active_arms = random.sample(range(N), M)
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
            alpha = 1.0 / counts[i][s][a]
            max_q_next = max(Q[i][next_s].values())
            Q[i][s][a] = (1 - alpha) * Q[i][s][a] + alpha * (r + max_q_next)
            lambda_est[i][s] = Q[i][s][1] - Q[i][s][0]

        cumulative_reward += step_reward / N
        X = X_next

    return cumulative_reward / T

def simulate_push_based(N, M, T, P_passive, P_active, state_rewards):
    """Push-based simulation"""
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

        cumulative_reward += step_reward / N
        X = X_next

    return cumulative_reward / T


# -----------------------

def run_experiments(N, M, T, num_runs, reward_type='focused', active_strength='perfect'):
    """
    Run experiments varying density (d) and reward scaling (alpha)
    Similar to the paper's approach with numerical parameters
    """
    # Define parameter ranges
    density_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # 5 levels
    alpha_values = [1.0, 2.0, 3.0,4.0,5.0]    # 5 levels
    
    #P_active = get_active_transition(active_strength)
    P_active = generate_active_transitions(beta=0.1, n=5, goal=2)


    
    results = []
    total_configs = len(density_values) * len(alpha_values)
    current = 0
    
    print(f"\nRunning {reward_type.upper()} reward experiment")
    print(f"Parameter grid: {len(density_values)} densities × {len(alpha_values)} alphas = {total_configs} configs")
    print("="*60)
    
    for density in density_values:
        for alpha in alpha_values:
            current += 1
            print(f"Config {current}/{total_configs}: d={density:.1f}, α={alpha:.2f}", end=' ... ')
            
            P_passive = generate_passive_transitions_smooth(
                density=density,
                n=5,
                sigma_min=0, sigma_max=1,
                p=1.0,
                cap_hops=2,   # set to None for full exponential tails
                ring=False
            )


            # Reward structure (assuming your version accepts these args)
            state_rewards = generate_reward_structure(alpha, reward_type)

            
            # Run multiple iterations
            pull_rewards = []
            push_rewards = []
            
            for run_id in range(num_runs):
                np.random.seed(42 + run_id)
                random.seed(42 + run_id)
                
                pull_final = simulate_pull_wiql(N, M, T, P_passive, P_active, state_rewards)
                push_final = simulate_push_based(N, M, T, P_passive, P_active, state_rewards)
                
                pull_rewards.append(pull_final)
                push_rewards.append(push_final)
            
            avg_pull = np.mean(pull_rewards)
            avg_push = np.mean(push_rewards)
            
            results.append({
                'density': density,
                'alpha': alpha,
                'pull_reward': avg_pull,
                'push_reward': avg_push,
                'difference': avg_pull - avg_push
            })
            
            print(f"Pull: {avg_pull:.3f}, Push: {avg_push:.3f}")
    
    return pd.DataFrame(results)

# -----------------------
# Plotting Functions
# -----------------------


def create_heatmaps(df_focused: pd.DataFrame, df_spread: pd.DataFrame, output_dir: str = "results"):
    """
    Create six heatmaps (Pull/Push/Diff × Focused/Spread) with consistent formatting.

    - Figure size: 6×4 inches
    - Font sizes: 16 pt throughout
    - High-resolution output (300 dpi)
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set(context="notebook")

    # Collect unique sorted axis values
    density_vals = sorted(df_focused["density"].unique())
    alpha_vals = sorted(df_focused["alpha"].unique())

    # Pivot the data for both reward types
    pivots = {}
    for reward_type, df in [("focused", df_focused), ("spread", df_spread)]:
        pivots[f"{reward_type}_pull"] = df.pivot(index="alpha", columns="density", values="pull_reward")
        pivots[f"{reward_type}_push"] = df.pivot(index="alpha", columns="density", values="push_reward")
        pivots[f"{reward_type}_diff"] = df.pivot(index="alpha", columns="density", values="difference")

    # Configurations (same as before)
    heatmap_configs = [
        ("pull_focused", pivots["focused_pull"], "Pull-Based (Focused)", "YlGnBu_r", "Avg Reward", None),
        ("push_focused", pivots["focused_push"], "Push-Based (Focused)", "YlGnBu_r", "Avg Reward", None),
        ("diff_focused", pivots["focused_diff"], "Pull Advantage (Focused)", "RdYlGn", "Pull Advantage", 0),
        ("pull_spread",  pivots["spread_pull"],  "Pull-Based (Spread)",  "YlGnBu_r", "Avg Reward", None),
        ("push_spread",  pivots["spread_push"],  "Push-Based (Spread)",  "YlGnBu_r", "Avg Reward", None),
        ("diff_spread",  pivots["spread_diff"],  "Pull Advantage (Spread)", "RdYlGn", "Pull Advantage", 0),
    ]

    # Consistent font sizes for paper-ready figures
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    })

    saved_files = []

    # Loop through and plot each heatmap
    for filename, pivot, title, cmap, cbar_label, center in heatmap_configs:
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            pivot,
            cmap=cmap,
            center=center,
            annot=False,
            fmt=".2f",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": cbar_label},
            ax=ax,
        )

        # Axis labels
        ax.set_xlabel("Density (d)", fontsize=16)
        ax.set_ylabel("Reward Scale (α)", fontsize=16)
        ax.invert_yaxis()

        # Optional: small spacing adjustment
        plt.tight_layout()

        # Save each file
        filepath_png = os.path.join(output_dir, f"{filename}.png")
        filepath_pdf = os.path.join(output_dir, f"{filename}.pdf")

        plt.savefig(filepath_png, dpi=300, bbox_inches="tight")
        plt.savefig(filepath_pdf, dpi=300, bbox_inches="tight")

        saved_files.append(filepath_png)
        saved_files.append(filepath_pdf)

        print(f"Saved: {filepath_png}\nSaved: {filepath_pdf}")
        plt.close(fig)

    print(f"\nAll 6 heatmaps saved to {output_dir}/")
    return saved_files


# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run numerical heatmap experiments')
    parser.add_argument('-N', '--num_arms', type=int, default=10,
                        help='Number of arms/nodes (default: 100)')
    parser.add_argument('-M', '--num_activations', type=int, default=2,
                        help='Number of activations per step (default: 10)')
    parser.add_argument('-T', '--time_steps', type=int, default=2000,
                        help='Number of time steps (default: 5000)')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs to average per config (default: 10)')
    parser.add_argument('--active', type=str, default='perfect',
                        choices=['perfect', 'strong', 'moderate'],
                        help='Active action strength (default: perfect)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("NUMERICAL HEATMAP EXPERIMENT: PULL vs PUSH")
    print("="*60)
    print(f"Parameters: N={args.num_arms}, M={args.num_activations}, T={args.time_steps}")
    print(f"Runs per config: {args.num_runs}")
    print(f"Active strength: {args.active}")
    print(f"Grid: 5 densities × 5 alphas = 25 configs per reward type")
    print(f"Total configs: 50 (25 focused + 25 spread)")
    print("="*60)
    
    # Run experiments for both reward types
    df_focused = run_experiments(args.num_arms, args.num_activations, args.time_steps,
                                  args.num_runs, reward_type='focused', 
                                  active_strength=args.active)
    
    df_spread = run_experiments(args.num_arms, args.num_activations, args.time_steps,
                                 args.num_runs, reward_type='spread',
                                 active_strength=args.active)
    
    # Save results
    import os
    os.makedirs('results', exist_ok=True)
    df_focused.to_csv('results/experiment_results_focused.csv', index=False)
    df_spread.to_csv('results/experiment_results_spread.csv', index=False)
    print("\nResults saved to results/experiment_results_*.csv")
    
    # Create heatmaps
    print("\nGenerating heatmaps...")
    create_heatmaps(df_focused, df_spread)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nFOCUSED REWARDS:")
    print(f"  Pull avg: {df_focused['pull_reward'].mean():.4f}")
    print(f"  Push avg: {df_focused['push_reward'].mean():.4f}")
    print(f"  Pull advantage: {df_focused['difference'].mean():.4f}")
    
    print("\nSPREAD REWARDS:")
    print(f"  Pull avg: {df_spread['pull_reward'].mean():.4f}")
    print(f"  Push avg: {df_spread['push_reward'].mean():.4f}")
    print(f"  Pull advantage: {df_spread['difference'].mean():.4f}")
    
    print("\nBest Pull advantage (focused):")
    best_focused = df_focused.loc[df_focused['difference'].idxmax()]
    print(f"  d={best_focused['density']:.1f}, α={best_focused['alpha']:.2f}: +{best_focused['difference']:.4f}")
    
    print("\nBest Pull advantage (spread):")
    best_spread = df_spread.loc[df_spread['difference'].idxmax()]
    print(f"  d={best_spread['density']:.1f}, α={best_spread['alpha']:.2f}: +{best_spread['difference']:.4f}")
    print("="*60)