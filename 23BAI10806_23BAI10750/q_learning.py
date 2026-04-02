"""
q_learning.py
=============
Tabular Q-Learning agent trained on the custom GridWorldEnv.

Hyper-parameters
----------------
  alpha   : learning rate
  gamma   : discount factor
  epsilon : exploration rate (decays over episodes)

Training
--------
  5 000 episodes with ε-greedy action selection.

Output
------
  - Average reward over last 100 episodes
  - Success rate (% of episodes that reached the goal)
  - Matplotlib plot: reward per episode
"""

import random
import matplotlib.pyplot as plt
from grid_world_env import GridWorldEnv   # Import the custom environment


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
N_STATES       = 36       # 6×6 grid  → 36 discrete states
N_ACTIONS      = 4        # Up / Down / Left / Right

ALPHA          = 0.1      # Learning rate
GAMMA          = 0.99     # Discount factor

EPSILON_START  = 1.0      # Initial exploration rate (fully random)
EPSILON_MIN    = 0.01     # Minimum exploration rate
EPSILON_DECAY  = 0.995    # Multiplicative decay applied each episode

N_EPISODES     = 5000     # Total training episodes


# ---------------------------------------------------------------------------
# Q-table initialisation
# ---------------------------------------------------------------------------
def build_q_table(n_states: int, n_actions: int):
    """Return a zero-initialised Q-table of shape (n_states, n_actions)."""
    return [[0.0] * n_actions for _ in range(n_states)]


# ---------------------------------------------------------------------------
# Action selection  (ε-greedy)
# ---------------------------------------------------------------------------
def select_action(q_table, state: int, epsilon: float) -> int:
    """
    Choose an action using the ε-greedy policy.

    With probability ε  → random action   (exploration)
    With probability 1-ε → greedy action  (exploitation)
    """
    if random.random() < epsilon:
        return random.randint(0, N_ACTIONS - 1)       # Explore
    return q_table[state].index(max(q_table[state]))  # Exploit


# ---------------------------------------------------------------------------
# Q-value update  (Bellman equation)
# ---------------------------------------------------------------------------
def update_q(q_table, state: int, action: int,
             reward: float, next_state: int, done: bool):
    """
    Apply the Q-learning update rule:

        Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') - Q(s, a)]

    If the episode is done, there is no future value (target = reward only).
    """
    best_next = max(q_table[next_state]) if not done else 0.0
    td_target = reward + GAMMA * best_next
    td_error  = td_target - q_table[state][action]
    q_table[state][action] += ALPHA * td_error


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train():
    """Train the Q-learning agent and return the episode reward history."""
    env      = GridWorldEnv()
    q_table  = build_q_table(N_STATES, N_ACTIONS)
    epsilon  = EPSILON_START

    episode_rewards  = []   # Total reward collected each episode
    success_count    = 0    # Episodes where agent reached the goal

    for episode in range(1, N_EPISODES + 1):

        state       = env.reset()
        total_reward = 0.0
        done        = False

        while not done:
            action                    = select_action(q_table, state, epsilon)
            next_state, reward, done  = env.step(action)
            update_q(q_table, state, action, reward, next_state, done)

            state        = next_state
            total_reward += reward

        # Track whether the agent reached the goal this episode
        if reward == 10:   # Goal reward is uniquely +10
            success_count += 1

        episode_rewards.append(total_reward)

        # Decay epsilon after every episode
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # Progress update every 500 episodes
        if episode % 500 == 0:
            avg = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
            print(f"  Episode {episode:5d} | "
                  f"Avg reward (last 100): {avg:7.2f} | "
                  f"ε = {epsilon:.4f}")

    return q_table, episode_rewards, success_count


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------
def print_summary(episode_rewards, success_count):
    """Print final training statistics."""
    avg_last_100  = sum(episode_rewards[-100:]) / 100
    success_rate  = (success_count / N_EPISODES) * 100

    print("\n" + "=" * 50)
    print("  Training Complete")
    print("=" * 50)
    print(f"  Total episodes         : {N_EPISODES}")
    print(f"  Average reward (last 100 eps) : {avg_last_100:.2f}")
    print(f"  Success rate           : {success_rate:.1f}%")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_rewards(episode_rewards):
    """
    Plot reward per episode and a 100-episode rolling average.
    Saves the figure as 'q_learning_rewards.png' and displays it.
    """
    # Compute 100-episode rolling average
    window      = 100
    rolling_avg = [
        sum(episode_rewards[max(0, i - window):i + 1]) /
        len(episode_rewards[max(0, i - window):i + 1])
        for i in range(len(episode_rewards))
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(episode_rewards, alpha=0.3, color="steelblue", linewidth=0.8,
            label="Reward per episode")
    ax.plot(rolling_avg,     color="crimson",   linewidth=2.0,
            label=f"{window}-episode rolling average")

    ax.set_title("Q-Learning: Reward vs Episode (GridWorld 6×6)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode",  fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig("q_learning_rewards.png", dpi=150)
    print("\n  Plot saved as 'q_learning_rewards.png'")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Q-Learning training on GridWorld 6×6 ...")
    print(f"  Episodes : {N_EPISODES}  |  α={ALPHA}  |  "
          f"γ={GAMMA}  |  ε₀={EPSILON_START} → {EPSILON_MIN}\n")

    q_table, episode_rewards, success_count = train()
    print_summary(episode_rewards, success_count)
    plot_rewards(episode_rewards)
