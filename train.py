"""
train.py
End-to-end training and evaluation script replicating Kolm & Ritter (2019).

Usage:
    python train.py                    # full run (paper defaults)
    python train.py --fast             # quick smoke-test (fewer episodes)
    python train.py --multiplier 0     # frictionless world
    python train.py --multiplier 5     # high friction (paper Exhibit 3)

Paper parameters:
    S0=100, K=100 (ATM), T=10 days, D=5 periods/day  => 50 steps/episode
    sigma=0.01 (daily log-vol), kappa=0.1 (risk aversion)
    5 batches × 15,000 episodes × 50 steps = 3,750,000 transitions
    (paper says 750,000 per batch = 15,000 × 50 ✓)
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from trading_env import OptionHedgingEnv, DeltaHedgingPolicy
from agent import FittedQAgent
from evaluate import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fast",       action="store_true", help="Quick test: 3 batches × 500 episodes")
    p.add_argument("--multiplier", type=float, default=5.0,
                   help="Transaction cost multiplier (0=frictionless, 5=high friction)")
    p.add_argument("--n_eval",     type=int,   default=10_000, help="Out-of-sample episodes")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out_dir",    type=str,   default="results")
    return p.parse_args()


def main():
    args = parse_args()

    # ---- Paper hyper-parameters ----
    S0    = 100.0
    K     = 100.0    # ATM
    T     = 10       # days
    D     = 5        # periods per day
    sigma = 0.01     # daily log-vol
    kappa = 0.1      # risk aversion

    if args.fast:
        num_batches        = 3
        episodes_per_batch = 500
        n_eval             = 200
        print("[FAST MODE] Using 3 batches × 500 episodes for quick testing. --n_eval ignored.")
    else:
        num_batches        = 5        # paper: 5 batches
        episodes_per_batch = 15_000   # paper: 750,000 / 50 = 15,000 episodes
        n_eval             = args.n_eval

    print("=" * 60)
    print("Kolm & Ritter (2019) — RL Option Hedging Replication")
    print("=" * 60)
    print(f"  S0={S0}, K={K}, T={T}d, D={D}/day => {T*D} steps/ep")
    print(f"  sigma={sigma}, kappa={kappa}")
    print(f"  multiplier={args.multiplier}, seed={args.seed}")
    print(f"  batches={num_batches}, episodes/batch={episodes_per_batch:,}")
    print(f"  total transitions={num_batches * episodes_per_batch * T * D:,}")
    print("=" * 60)

    # ---- Environments ----
    env_friction = OptionHedgingEnv(
        S0=S0, K=K, T=T, D=D, sigma=sigma, kappa=kappa,
        multiplier=args.multiplier,
    )
    env_frictionless = OptionHedgingEnv(
        S0=S0, K=K, T=T, D=D, sigma=sigma, kappa=kappa,
        multiplier=0.0,
    )

    # ---- Baseline delta-hedging policy ----
    delta_policy = DeltaHedgingPolicy(K=K, sigma=sigma)

    # ---- RL agent: Fitted Q-iteration with neural network ----
    agent = FittedQAgent(
        env               = env_friction,
        state_dim         = 3,
        action_dim        = 1,
        hidden_sizes      = [64, 64],
        gamma             = 1.0,           # finite-horizon, no discounting
        lr                = 1e-3,
        batch_size        = 2048,
        epochs_per_fit    = 10,
        num_batches       = num_batches,
        episodes_per_batch= episodes_per_batch,
        epsilon_start     = 1.0,
        epsilon_end       = 0.05,
        seed              = args.seed,
    )

    # ---- Train ----
    print("\n[Training RL agent...]")
    agent.train(verbose=True)

    # ---- Save model ----
    os.makedirs(args.out_dir, exist_ok=True)
    agent.save(f"{args.out_dir}/q_network.pt")

    # ---- Evaluate & plot ----
    print("\n[Evaluating...]")
    results = evaluate(
        env_frictionless = env_frictionless,
        env_friction     = env_friction,
        rl_agent         = agent,
        delta_policy     = delta_policy,
        n_eval           = n_eval,
        out_dir          = args.out_dir,
    )

    print("\n[Summary]")
    print(f"  RL mean cost:    {results['rl_costs'].mean():.4f}")
    print(f"  Delta mean cost: {results['delta_costs'].mean():.4f}")
    print(f"  Cost reduction:  {(1 - results['rl_costs'].mean()/results['delta_costs'].mean())*100:.1f}%")
    print(f"  Cost t-stat: {results['t_cost']:.2f}  (paper: -143.22)")


if __name__ == "__main__":
    main()
