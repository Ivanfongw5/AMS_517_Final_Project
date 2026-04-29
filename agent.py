"""
agent.py
Fitted Q-Iteration agent replicating Kolm & Ritter (2019).

The paper uses a nonlinear regression learner (neural network) fit to
SARSA targets derived from the Bellman equation. We implement:

  1. QNetwork     — MLP approximating q(s, a)
  2. ReplayBuffer — stores (s, a, r, s', a') tuples for batch fitting
  3. FittedQAgent — orchestrates batch generation + model fitting
                    using epsilon-greedy exploration with decay.

Training procedure (paper, p.162):
  - B = 5 batches, each with 750,000 (X, Y) pairs
  - Each episode has D*T = 50 steps
  - => 15,000 episodes per batch
  - After each batch, refit q-network to new SARSA targets
  - Epsilon decays across batches
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    MLP approximating q(s, a).
    Input:  concatenation of state (3-dim) and action (1-dim)  => 4-dim
    Output: scalar Q-value
    """

    def __init__(self, state_dim: int = 3, action_dim: int = 1, hidden_sizes: List[int] = [64, 64]):
        super().__init__()
        layers = []
        in_dim = state_dim + action_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Replay Buffer  (stores full SARSA tuples)
# ---------------------------------------------------------------------------

class SARSABuffer:
    """Stores (s, a, r, s', a') tuples."""

    def __init__(self):
        self.states:      List[np.ndarray] = []
        self.actions:     List[np.ndarray] = []
        self.rewards:     List[float]      = []
        self.next_states: List[np.ndarray] = []
        self.next_actions:List[np.ndarray] = []
        self.dones:       List[bool]       = []

    def push(self, s, a, r, s_, a_, done):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(s_)
        self.next_actions.append(a_)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.states       = []
        self.actions      = []
        self.rewards      = []
        self.next_states  = []
        self.next_actions = []
        self.dones        = []

    def as_tensors(self, device: torch.device):
        S  = torch.tensor(np.array(self.states),       dtype=torch.float32, device=device)
        A  = torch.tensor(np.array(self.actions),      dtype=torch.float32, device=device)
        R  = torch.tensor(np.array(self.rewards),      dtype=torch.float32, device=device)
        S_ = torch.tensor(np.array(self.next_states),  dtype=torch.float32, device=device)
        A_ = torch.tensor(np.array(self.next_actions), dtype=torch.float32, device=device)
        D  = torch.tensor(np.array(self.dones),        dtype=torch.float32, device=device)
        return S, A, R, S_, A_, D


# ---------------------------------------------------------------------------
# Fitted Q-Iteration Agent
# ---------------------------------------------------------------------------

class FittedQAgent:
    """
    Implements the batch fitted-Q / SARSA training loop from Kolm & Ritter (2019).

    Key hyper-parameters (paper defaults):
      num_batches        = 5
      episodes_per_batch = 15_000
      gamma              = 1.0  (finite horizon, no discounting needed; paper uses ~1)
      epsilon_start      = 1.0
      epsilon_end        = 0.05
    """

    def __init__(
        self,
        env,
        state_dim:          int   = 3,
        action_dim:         int   = 1,
        hidden_sizes:       List[int] = [64, 64],
        gamma:              float = 1.0,
        lr:                 float = 1e-3,
        batch_size:         int   = 2048,
        epochs_per_fit:     int   = 10,
        num_batches:        int   = 5,
        episodes_per_batch: int   = 15_000,
        epsilon_start:      float = 1.0,
        epsilon_end:        float = 0.05,
        device:             Optional[str] = None,
        seed:               Optional[int] = 42,
    ):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.epochs_per_fit = epochs_per_fit
        self.num_batches = num_batches
        self.episodes_per_batch = episodes_per_batch
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.buffer = SARSABuffer()
        self.action_space = env.action_space
        self.max_trade = int(env.max_trade)

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.training_losses: List[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _greedy_action(self, state: np.ndarray) -> np.ndarray:
        """Choose action that maximises Q(s,·) by grid search over integers."""
        candidates = np.arange(-self.max_trade, self.max_trade + 1, dtype=np.float32)
        s_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        s_rep = s_tensor.repeat(len(candidates), 1)
        a_tensor = torch.tensor(candidates, dtype=torch.float32, device=self.device).unsqueeze(1)
        self.q_net.eval()
        with torch.no_grad():
            q_vals = self.q_net(s_rep, a_tensor).cpu().numpy()
        self.q_net.train()
        best_idx = int(np.argmax(q_vals))
        return np.array([candidates[best_idx]], dtype=np.float32)

    def select_action(self, state: np.ndarray, epsilon: float) -> np.ndarray:
        if np.random.rand() < epsilon:
            # Random integer trade
            trade = np.random.randint(-self.max_trade, self.max_trade + 1)
            return np.array([float(trade)], dtype=np.float32)
        return self._greedy_action(state)

    # ------------------------------------------------------------------
    # SARSA target (Eq. 6 of paper)
    # ------------------------------------------------------------------

    def _compute_targets(self, S, A, R, S_, A_, D) -> torch.Tensor:
        """Y_t = r_{t+1} + gamma * q_hat(s_{t+1}, a_{t+1})  (Eq. 6).
        Targets are computed once per batch from the current q_net and then
        detached before fitting, so the network is not chasing a moving target
        within a batch. This is the correct fitted-Q iteration procedure.
        """
        self.q_net.eval()
        with torch.no_grad():
            q_next = self.q_net(S_, A_)
        self.q_net.train()
        targets = R + self.gamma * q_next * (1.0 - D)
        return targets

    # ------------------------------------------------------------------
    # Fit network to one batch
    # ------------------------------------------------------------------

    def _fit_batch(self):
        S, A, R, S_, A_, D = self.buffer.as_tensors(self.device)
        targets = self._compute_targets(S, A, R, S_, A_, D).detach()

        dataset = TensorDataset(S, A, targets)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        epoch_losses = []
        for _ in range(self.epochs_per_fit):
            for s_b, a_b, y_b in loader:
                self.optimizer.zero_grad()
                q_pred = self.q_net(s_b, a_b)
                loss = self.loss_fn(q_pred, y_b)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses))
        self.training_losses.append(avg_loss)
        return avg_loss

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, verbose: bool = True):
        """
        Run B batches of fitted-Q iteration as described in the paper.
        Each batch: collect episodes_per_batch episodes, then refit q_net.
        """
        epsilons = np.linspace(self.epsilon_start, self.epsilon_end, self.num_batches)

        for b in range(self.num_batches):
            eps = epsilons[b]
            self.buffer.clear()

            # ---- Collect episodes ----
            total_reward = 0.0
            for ep in range(self.episodes_per_batch):
                obs, _ = self.env.reset()
                action = self.select_action(obs, eps)
                ep_reward = 0.0

                while True:
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    next_action = self.select_action(next_obs, eps)
                    self.buffer.push(obs, action, reward, next_obs, next_action, float(done))
                    obs = next_obs
                    action = next_action
                    ep_reward += reward
                    if done:
                        break

                total_reward += ep_reward

            avg_ep_reward = total_reward / self.episodes_per_batch

            # ---- Fit Q-network ----
            loss = self._fit_batch()

            if verbose:
                print(
                    f"Batch {b+1}/{self.num_batches} | "
                    f"eps={eps:.3f} | "
                    f"buffer={len(self.buffer):,} | "
                    f"avg_ep_reward={avg_ep_reward:.4f} | "
                    f"fit_loss={loss:.6f}"
                )

        if verbose:
            print("Training complete.")

    # ------------------------------------------------------------------
    # Greedy policy inference (no exploration)
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> np.ndarray:
        return self._greedy_action(state)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        print(f"Model loaded from {path}")
