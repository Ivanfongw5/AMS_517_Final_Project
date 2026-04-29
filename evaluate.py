"""
evaluate.py
Out-of-sample evaluation replicating Exhibits 1–5 of Kolm & Ritter (2019).

Runs N=10,000 out-of-sample episodes for both:
  - Delta-hedging baseline  (pi_DH, Eq. 12)
  - RL agent               (greedy policy from fitted Q-network)

Produces:
  - Exhibit 1/3 style: single episode trace (stock pnl, option pnl, total pnl, position)
  - Exhibit 4 style:   KDE of total cost and realized vol of total P&L
  - Exhibit 5 style:   KDE of t-statistic of total P&L
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from typing import List, Dict
from trading_env import bs_delta


# ---------------------------------------------------------------------------
# Run one out-of-sample episode and collect per-step info
# ---------------------------------------------------------------------------

def run_episode(env, policy_fn, seed: int = None) -> Dict:
    """
    policy_fn(obs) -> action array
    Returns dict of per-step quantities.
    """
    obs, _ = env.reset(seed=seed)
    steps, stock_pnl_hist, option_pnl_hist, cost_hist = [], [], [], []
    n_hist, delta_hist, total_pnl_hist = [], [], []
    cumulative_total = 0.0
    step = 0

    while True:
        action = policy_fn(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        cumulative_total += info["dw"]
        delta = bs_delta(info["S"], env.K, info["tau"], env.sigma)

        steps.append(step)
        stock_pnl_hist.append(info["stock_pnl"])
        option_pnl_hist.append(info["option_pnl"])
        cost_hist.append(info["cost"])
        total_pnl_hist.append(cumulative_total)
        n_hist.append(info["n"])
        delta_hist.append(-env.shares_per_contract * delta)

        obs = next_obs
        step += 1
        if done:
            break

    return {
        "steps":          np.array(steps),
        "stock_pnl":      np.cumsum(stock_pnl_hist),
        "option_pnl":     np.cumsum(option_pnl_hist),
        "cost_pnl":       -np.cumsum(cost_hist),
        "total_pnl":      np.array(total_pnl_hist),
        "position":       np.array(n_hist),
        "delta_position": np.array(delta_hist),
        "total_cost":     float(np.sum(cost_hist)),
        "realized_vol":   float(np.std(np.diff(total_pnl_hist))) if len(total_pnl_hist) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Exhibit 1 / 3  — single path trace
# ---------------------------------------------------------------------------

def plot_single_episode(episode: Dict, title: str, save_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    t = episode["steps"]

    ax = axes[0]
    ax.plot(t, episode["stock_pnl"],  label="stock pnl",  color="steelblue")
    ax.plot(t, episode["option_pnl"], label="option pnl", color="tomato")
    ax.plot(t, episode["total_pnl"],  label="total pnl",  color="black", linewidth=2)
    ax.plot(t, episode["cost_pnl"],   label="cost pnl",   color="orange", linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel("Value (dollars)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, episode["position"],       label="stock.pos.shares",  color="steelblue")
    ax.plot(t, episode["delta_position"], label="delta.hedge.shares", color="tomato",
            linestyle="--", alpha=0.7)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Timestep (D*T)")
    ax.set_ylabel("Shares")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Exhibit 4  — KDE of total cost and realized vol
# ---------------------------------------------------------------------------

def plot_kde_cost_vol(
    delta_costs: np.ndarray, rl_costs: np.ndarray,
    delta_vols:  np.ndarray, rl_vols:  np.ndarray,
    save_path: str,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for arr, label, color in [(delta_costs, "Delta", "steelblue"), (rl_costs, "Reinf", "tomato")]:
        kde = stats.gaussian_kde(arr)
        xs = np.linspace(arr.min(), arr.max(), 300)
        axes[0].plot(xs, kde(xs), label=label, color=color)
    axes[0].set_xlabel("Total cost")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Exhibit 4 (left): Total Cost KDE")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    for arr, label, color in [(delta_vols, "Delta", "steelblue"), (rl_vols, "Reinf", "tomato")]:
        kde = stats.gaussian_kde(arr)
        xs = np.linspace(arr.min(), arr.max(), 300)
        axes[1].plot(xs, kde(xs), label=label, color=color)
    axes[1].set_xlabel("Realized vol")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Exhibit 4 (right): Realized Vol KDE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Paired t-tests: delta and RL episodes share the same GBM seed per iteration,
    # so samples are paired (same price path, different policy). ttest_rel is correct.
    t_cost, p_cost = stats.ttest_rel(delta_costs, rl_costs)
    t_vol,  p_vol  = stats.ttest_rel(delta_vols,  rl_vols)
    fig.suptitle(
        f"Cost t-stat={t_cost:.2f} (p={p_cost:.2e}) | "
        f"Vol t-stat={t_vol:.2f} (p={p_vol:.2e})",
        fontsize=10,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave room for suptitle
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
    return t_cost, p_cost, t_vol, p_vol


# ---------------------------------------------------------------------------
# Exhibit 5  — KDE of t-statistic of total P&L
# ---------------------------------------------------------------------------

def compute_pnl_t_stats(all_pnls: List[np.ndarray]) -> np.ndarray:
    """For each episode, compute the t-statistic of the total P&L sequence."""
    t_stats = []
    for pnl in all_pnls:
        if len(pnl) > 1:
            t_stat, _ = stats.ttest_1samp(np.diff(pnl), 0)
            t_stats.append(t_stat)
    return np.array(t_stats)


def plot_pnl_t_stats(delta_t: np.ndarray, rl_t: np.ndarray, save_path: str):
    fig, ax = plt.subplots(figsize=(9, 5))

    for arr, label, color in [(delta_t, "Delta", "steelblue"), (rl_t, "Reinf", "tomato")]:
        finite = arr[np.isfinite(arr)]
        kde = stats.gaussian_kde(finite)
        xs = np.linspace(finite.min(), finite.max(), 300)
        ax.plot(xs, kde(xs), label=label, color=color)

    ax.set_xlabel("Student t-statistic of total P&L")
    ax.set_ylabel("Density")
    ax.set_title("Exhibit 5: KDE of t-statistic of Total P&L per Episode")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def evaluate(
    env_frictionless,
    env_friction,
    rl_agent,
    delta_policy,
    n_eval: int = 10_000,
    out_dir: str = ".",
):
    import os
    os.makedirs(out_dir, exist_ok=True)

    # ---- Exhibit 1: single frictionless RL episode ----
    ep = run_episode(env_frictionless, rl_agent.act, seed=0)
    plot_single_episode(ep, "Exhibit 1: RL Agent — Frictionless World",
                        f"{out_dir}/exhibit1_rl_frictionless.png")

    # ---- Exhibit 2: single friction delta episode ----
    ep = run_episode(env_friction, delta_policy, seed=0)
    plot_single_episode(ep, "Exhibit 2: Delta-Hedging Baseline — High Friction",
                        f"{out_dir}/exhibit2_delta_friction.png")

    # ---- Exhibit 3: single friction RL episode ----
    ep = run_episode(env_friction, rl_agent.act, seed=0)
    plot_single_episode(ep, "Exhibit 3: RL Agent — High Friction (cost-conscious)",
                        f"{out_dir}/exhibit3_rl_friction.png")

    # ---- Exhibit 4 & 5: N=10,000 out-of-sample runs ----
    print(f"\nRunning {n_eval:,} out-of-sample episodes for Exhibits 4 & 5 ...")

    delta_costs, rl_costs = [], []
    delta_vols,  rl_vols  = [], []
    delta_pnls,  rl_pnls  = [], []

    for i in range(n_eval):
        seed = 1000 + i
        d_ep = run_episode(env_friction, delta_policy, seed=seed)
        r_ep = run_episode(env_friction, rl_agent.act,  seed=seed)

        delta_costs.append(d_ep["total_cost"])
        rl_costs.append(r_ep["total_cost"])
        delta_vols.append(d_ep["realized_vol"])
        rl_vols.append(r_ep["realized_vol"])
        delta_pnls.append(d_ep["total_pnl"])
        rl_pnls.append(r_ep["total_pnl"])

        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{n_eval:,} done")

    delta_costs = np.array(delta_costs)
    rl_costs    = np.array(rl_costs)
    delta_vols  = np.array(delta_vols)
    rl_vols     = np.array(rl_vols)

    t_cost, p_cost, t_vol, p_vol = plot_kde_cost_vol(
        delta_costs, rl_costs, delta_vols, rl_vols,
        f"{out_dir}/exhibit4_kde_cost_vol.png"
    )

    print(f"\nExhibit 4 results:")
    print(f"  Cost: delta_mean={delta_costs.mean():.4f}, rl_mean={rl_costs.mean():.4f}")
    print(f"  Cost t-stat={t_cost:.2f}, p={p_cost:.2e}")
    print(f"  Vol:  delta_mean={delta_vols.mean():.6f}, rl_mean={rl_vols.mean():.6f}")
    print(f"  Vol  t-stat={t_vol:.2f}, p={p_vol:.2e}")

    delta_t = compute_pnl_t_stats(delta_pnls)
    rl_t    = compute_pnl_t_stats(rl_pnls)
    plot_pnl_t_stats(delta_t, rl_t, f"{out_dir}/exhibit5_pnl_t_stats.png")

    print("\nAll exhibits saved.")
    return {
        "delta_costs": delta_costs, "rl_costs": rl_costs,
        "delta_vols": delta_vols,   "rl_vols": rl_vols,
        "t_cost": t_cost, "p_cost": p_cost,
        "t_vol": t_vol,   "p_vol": p_vol,
    }
