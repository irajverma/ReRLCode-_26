"""
dqn.py
======
Deep Q-Network (DQN) agent trained on the custom GridWorldEnv.

Architecture
------------
  Input  : one-hot vector of size 36  (one entry per grid cell)
  Hidden : Linear(36, 64) → ReLU → Linear(64, 64) → ReLU
  Output : Linear(64, 4)  → 4 Q-values, one per action

Key components
--------------
  ReplayBuffer   — stores (s, a, r, s', done) transitions
  DQNetwork      — the online Q-network  (trained every step)
  TargetDQNetwork— copy of DQNetwork updated every TARGET_UPDATE steps
  DQNAgent       — orchestrates action selection, memory, and learning
  train()        — 5 000-episode training loop
  plot_rewards() — matplotlib reward-vs-episode curve
"""

import random
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from grid_world_env import GridWorldEnv   # Custom 6×6 environment


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
N_STATES        = 36          # One-hot input dimension  (6×6 grid)
N_ACTIONS       = 4           # Up / Down / Left / Right

HIDDEN_SIZE     = 64          # Neurons in each hidden layer
LEARNING_RATE   = 1e-3        # Adam learning rate
GAMMA           = 0.99        # Discount factor

EPSILON_START   = 1.0         # Initial exploration rate
EPSILON_MIN     = 0.01        # Minimum exploration rate
EPSILON_DECAY   = 0.995       # Per-episode multiplicative decay

BUFFER_CAPACITY = 10_000      # Maximum transitions stored in replay buffer
BATCH_SIZE      = 64          # Mini-batch size for each gradient update
TARGET_UPDATE   = 100         # Update target network every N steps

N_EPISODES      = 5000        # Total training episodes

# Device: use GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# One-hot encoding utility
# ---------------------------------------------------------------------------
def one_hot(state: int, n_states: int = N_STATES) -> torch.Tensor:
    """
    Convert an integer state index into a one-hot float tensor.

    Example: state=3, n_states=36  →  tensor([0,0,0,1,0,...,0])

    Returns
    -------
    torch.Tensor of shape (1, n_states) on DEVICE
    """
    vec = torch.zeros(1, n_states, device=DEVICE)
    vec[0, state] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Neural network  (shared architecture for online & target nets)
# ---------------------------------------------------------------------------
class DQNetwork(nn.Module):
    """
    Fully connected Q-network.

    Layers:  36 → 64 → 64 → 4
    """

    def __init__(self, n_states: int, hidden: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x shape (batch, n_states) → (batch, n_actions)."""
        return self.net(x)


# ---------------------------------------------------------------------------
# Experience Replay Buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """
    Circular buffer that stores experience tuples
    (state, action, reward, next_state, done).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to keep.
    """

    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state: int, action: int, reward: float,
             next_state: int, done: bool):
        """Add a single transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """
        Randomly sample a mini-batch of transitions.

        Returns five separate lists: states, actions, rewards,
        next_states, dones — each of length batch_size.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    """
    Encapsulates the online network, target network, replay buffer,
    optimizer, and all learning logic.

    Parameters
    ----------
    n_states  : int   — size of the one-hot state vector
    n_actions : int   — number of discrete actions
    """

    def __init__(self, n_states: int, n_actions: int):
        # Online network — trained every step via gradient descent
        self.online_net = DQNetwork(n_states, HIDDEN_SIZE, n_actions).to(DEVICE)

        # Target network — periodically synced with online_net
        self.target_net = DQNetwork(n_states, HIDDEN_SIZE, n_actions).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()                  # Target net is never trained directly

        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn    = nn.MSELoss()

        self.memory     = ReplayBuffer(BUFFER_CAPACITY)
        self.step_count = 0                     # Global step counter (for target sync)

    # ------------------------------------------------------------------
    # Action selection  (ε-greedy)
    # ------------------------------------------------------------------
    def select_action(self, state: int, epsilon: float) -> int:
        """
        ε-greedy policy: explore randomly with probability ε,
        otherwise pick the action with the highest predicted Q-value.
        """
        if random.random() < epsilon:
            return random.randint(0, N_ACTIONS - 1)

        with torch.no_grad():
            q_values = self.online_net(one_hot(state))   # (1, 4)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Store transition
    # ------------------------------------------------------------------
    def store(self, state: int, action: int, reward: float,
              next_state: int, done: bool):
        """Push one transition into the replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------
    def learn(self):
        """
        Sample a mini-batch from the replay buffer and perform one
        gradient-descent step on the online network using the Bellman
        target computed from the (frozen) target network.

        Skips learning until the buffer has at least BATCH_SIZE samples.
        """
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Convert to tensors
        state_batch      = torch.cat([one_hot(s) for s in states])          # (B, 36)
        next_state_batch = torch.cat([one_hot(s) for s in next_states])     # (B, 36)
        action_batch     = torch.tensor(actions,  device=DEVICE).unsqueeze(1)  # (B, 1)
        reward_batch     = torch.tensor(rewards,  device=DEVICE, dtype=torch.float32)
        done_batch       = torch.tensor(dones,    device=DEVICE, dtype=torch.float32)

        # Q(s, a) from online network — only the taken action's Q-value
        q_current = self.online_net(state_batch).gather(1, action_batch).squeeze(1)

        # max_a' Q_target(s', a')  — from frozen target network
        with torch.no_grad():
            q_next = self.target_net(next_state_batch).max(dim=1).values

        # Bellman target:  r  +  γ · max Q_target(s')  ·  (1 - done)
        q_target = reward_batch + GAMMA * q_next * (1.0 - done_batch)

        # Gradient descent on MSE loss
        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Increment global step and sync target network periodically
        self.step_count += 1
        if self.step_count % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train():
    """
    Train the DQN agent for N_EPISODES episodes.

    Returns
    -------
    episode_rewards : list[float]  — total reward per episode
    success_count   : int          — episodes where goal was reached
    """
    env     = GridWorldEnv()
    agent   = DQNAgent(N_STATES, N_ACTIONS)
    epsilon = EPSILON_START

    episode_rewards = []
    success_count   = 0

    for episode in range(1, N_EPISODES + 1):

        state        = env.reset()
        total_reward = 0.0
        done         = False

        while not done:
            action                    = agent.select_action(state, epsilon)
            next_state, reward, done  = env.step(action)

            agent.store(state, action, reward, next_state, done)
            agent.learn()

            state        = next_state
            total_reward += reward

        # Track success (goal reward is uniquely +10)
        if reward == 10:
            success_count += 1

        episode_rewards.append(total_reward)

        # Decay epsilon every episode
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Progress log every 500 episodes
        if episode % 500 == 0:
            avg = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            print(f"  Episode {episode:5d} | "
                  f"Avg reward (last 100): {avg:7.2f} | "
                  f"ε = {epsilon:.4f} | "
                  f"Steps: {agent.step_count}")

    return episode_rewards, success_count


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------
def print_summary(episode_rewards: list, success_count: int):
    """Print final training statistics to console."""
    avg_last_100 = sum(episode_rewards[-100:]) / 100
    success_rate = (success_count / N_EPISODES) * 100

    print("\n" + "=" * 55)
    print("  DQN Training Complete")
    print("=" * 55)
    print(f"  Total episodes              : {N_EPISODES}")
    print(f"  Average reward (last 100)   : {avg_last_100:.2f}")
    print(f"  Success rate                : {success_rate:.1f}%")
    print(f"  Device used                 : {DEVICE}")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_rewards(episode_rewards: list):
    """
    Plot total reward per episode and a 100-episode rolling average.
    Saves the figure as 'dqn_rewards.png' and opens it.
    """
    window      = 100
    rolling_avg = [
        sum(episode_rewards[max(0, i - window): i + 1]) /
        len(episode_rewards[max(0, i - window): i + 1])
        for i in range(len(episode_rewards))
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episode_rewards, alpha=0.3, color="royalblue",
            linewidth=0.8, label="Reward per episode")
    ax.plot(rolling_avg, color="darkorange", linewidth=2.2,
            label=f"{window}-episode rolling average")

    ax.set_title("DQN: Reward vs Episode (GridWorld 6×6)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode",      fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("dqn_rewards.png", dpi=150)
    print("\n  Plot saved as 'dqn_rewards.png'")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting DQN training on GridWorld 6×6 ...")
    print(f"  Episodes : {N_EPISODES}  |  lr={LEARNING_RATE}  |  "
          f"γ={GAMMA}  |  batch={BATCH_SIZE}  |  "
          f"target_update={TARGET_UPDATE}  |  device={DEVICE}\n")

    episode_rewards, success_count = train()
    print_summary(episode_rewards, success_count)
    plot_rewards(episode_rewards)
