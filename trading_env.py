"""
trading_env.py
Gymnasium-compatible environment replicating Kolm & Ritter (2019):
  "Dynamic Replication and Hedging: A Reinforcement Learning Approach"

State:  (S_t, tau, n_t)  — stock price, time-to-expiry, current share holding
Action: integer shares to trade in [-max_trade, +max_trade]
Reward: mean-variance  R_t = dw_t - (kappa/2) * dw_t^2
        where dw_t = q_t - cost(trade)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.stats import norm
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def bs_call_price(S: float, K: float, tau: float, sigma: float, r: float = 0.0) -> float:
    """European call price under BSM.
    tau   : time to expiry in DAYS
    sigma : volatility in DAILY units  (same time unit as tau)
    """
    if tau <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)


def bs_delta(S: float, K: float, tau: float, sigma: float, r: float = 0.0) -> float:
    """BSM delta of a European call.
    tau   : time to expiry in DAYS
    sigma : volatility in DAILY units  (same time unit as tau)
    """
    if tau <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


# ---------------------------------------------------------------------------
# Transaction cost  (Eq. 13 of the paper)
# ---------------------------------------------------------------------------

def transaction_cost(trade: int, multiplier: float = 1.0, tick_size: float = 0.1) -> float:
    """
    cost(n) = multiplier * (tick_size * |n| + 0.01 * n^2)
    multiplier=0  → frictionless world (Exhibit 1)
    multiplier=5  → high friction  (Exhibit 3)
    """
    return multiplier * (tick_size * abs(trade) + 0.01 * trade ** 2)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OptionHedgingEnv(gym.Env):
    """
    Replicates the BSM simulation environment from Kolm & Ritter (2019).

    Parameters
    ----------
    S0        : initial stock price          (paper: 100)
    K         : strike price                 (paper: S0, ATM)
    T         : days to maturity             (paper: 10)
    D         : periods per day              (paper: 5)
    sigma     : daily log-vol                (paper: 0.01)
    kappa     : risk-aversion                (paper: 0.1)
    multiplier: trading-cost multiplier      (paper: 0 or 5)
    L         : number of option contracts   (paper: 1)
    shares_per_contract : shares per contract (paper: 100)
    max_trade : max shares traded per step   (paper: 100)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        S0: float = 100.0,
        K: Optional[float] = None,
        T: int = 10,
        D: int = 5,
        sigma: float = 0.01,
        kappa: float = 0.1,
        multiplier: float = 1.0,
        L: int = 1,
        shares_per_contract: int = 100,
        max_trade: int = 100,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.S0 = S0
        self.K = K if K is not None else S0   # ATM by default
        self.T = T
        self.D = D
        self.sigma = sigma          # daily vol
        self.sigma_dt = sigma / np.sqrt(D)   # per-period vol
        self.dt = 1.0 / D           # fraction of a day per step
        self.total_steps = T * D    # 50 by default
        self.kappa = kappa
        self.multiplier = multiplier
        self.L = L
        self.shares_per_contract = shares_per_contract
        self.max_trade = max_trade
        self.render_mode = render_mode

        # State: (S_t, tau_t, n_t)  — 3-dimensional continuous state
        # We normalise for the neural network but keep raw values for stepping
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -float(max_trade)], dtype=np.float32),
            high=np.array([np.inf, float(T), float(max_trade)], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: integer shares to trade in [-max_trade, max_trade]
        # We use a Discrete space mapped to integers for cleanliness,
        # but wrap it so the agent sees a Box [-1,1] if desired.
        # Here we expose the integer-valued Box directly.
        self.action_space = spaces.Box(
            low=np.array([-float(max_trade)], dtype=np.float32),
            high=np.array([float(max_trade)], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self._step: int = 0
        self._S: float = 0.0
        self._n: int = 0          # current share holding (integer)
        self._option_value_prev: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        self._step = 0
        self._S = self.S0
        self._n = 0

        # Option value at t=0 (tau in days, same unit as sigma)
        tau0 = float(self.T)
        self._option_value_prev = bs_call_price(self._S, self.K, tau0, self.sigma)

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Clip and round requested trade to integer shares
        trade_requested = int(np.clip(np.round(action[0]), -self.max_trade, self.max_trade))

        # Compute actual executed trade after position limits.
        # Cost is charged on shares actually traded, not the requested amount.
        n_new = int(np.clip(self._n + trade_requested, -self.max_trade, self.max_trade))
        trade_executed = n_new - self._n

        # --- Stock P&L from holding n shares through this period ---
        S_prev = self._S

        # GBM step
        z = self.np_random.standard_normal()
        self._S = S_prev * np.exp(-0.5 * self.sigma_dt ** 2 + self.sigma_dt * z)

        self._step += 1
        tau_new = (self.total_steps - self._step) * self.dt

        # New option value
        option_value_new = bs_call_price(self._S, self.K, tau_new, self.sigma)

        # --- Wealth increment (Eq. in paper, Section "Automatic Hedging in Theory") ---
        # dw_t = stock_pnl + option_pnl - cost
        # stock_pnl  = n_prev * (S_new - S_prev)   [holding before the trade]
        # option_pnl = L * shares_per_contract * (V_new - V_prev)  [long option]
        # cost       = transaction_cost on EXECUTED shares (not the requested amount)
        stock_pnl = self._n * (self._S - S_prev)
        option_pnl = self.L * self.shares_per_contract * (option_value_new - self._option_value_prev)
        cost = transaction_cost(trade_executed, self.multiplier)

        dw = stock_pnl + option_pnl - cost

        # Update position after computing pnl (trade settles end of period)
        self._n = n_new
        self._option_value_prev = option_value_new

        # --- Mean-variance reward (Eq. 10 in paper) ---
        reward = float(dw - (self.kappa / 2.0) * dw ** 2)

        terminated = self._step >= self.total_steps
        truncated = False

        info = {
            "S": self._S,
            "tau": tau_new,
            "n": self._n,
            "trade": trade_executed,
            "stock_pnl": stock_pnl,
            "option_pnl": option_pnl,
            "cost": cost,
            "dw": dw,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        tau = (self.total_steps - self._step) * self.dt
        return np.array([self._S, tau, float(self._n)], dtype=np.float32)

    def render(self):
        tau = (self.total_steps - self._step) * self.dt
        print(f"Step {self._step:3d} | S={self._S:.4f} | tau={tau:.4f} | n={self._n}")


# ---------------------------------------------------------------------------
# Baseline: Delta-hedging policy  (Eq. 12 of the paper)
# ---------------------------------------------------------------------------

class DeltaHedgingPolicy:
    """
    pi_DH: trade to match BSM delta each period.
    target_shares = -round(100 * delta(S, tau))
    action = target_shares - current_n
    """

    def __init__(self, K: float, sigma: float, shares_per_contract: int = 100, max_trade: int = 100):
        self.K = K
        self.sigma = sigma
        self.shares_per_contract = shares_per_contract
        self.max_trade = max_trade

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        S, tau, n = float(obs[0]), float(obs[1]), float(obs[2])
        delta = bs_delta(S, self.K, tau, self.sigma)
        target = -round(self.shares_per_contract * delta)
        trade = target - int(n)
        # Clip to match what the environment will actually execute
        trade = int(np.clip(trade, -self.max_trade, self.max_trade))
        return np.array([float(trade)], dtype=np.float32)
