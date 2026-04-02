# Reinforcement Learning Lab Project Report

**Course:** Reinforcement Learning (Sem 8)
**Title:** Comparative Study of Q-Learning and Deep Q-Network on a Custom Grid World Environment
**Total Marks:** 35

---

## Abstract

This report presents the design and implementation of a custom 6×6 Grid World environment and the training of two reinforcement learning agents — Tabular Q-Learning and Deep Q-Network (DQN) — on this environment. The environment features a start state, a goal state, two static obstacles, and one randomly-moving dynamic obstacle. Both agents were trained for 5,000 episodes. Q-Learning uses a tabular state-action value function updated via the Bellman equation, while DQN approximates the Q-function using a fully-connected neural network with experience replay and a target network. Experimental results show that both agents converge to near-optimal average rewards (≈ 0 to +1), with Q-Learning converging faster and exhibiting greater stability, while DQN demonstrates stronger generalizability. The dynamic obstacle introduces stochastic perturbations that prevent both agents from achieving a perfectly consistent optimal policy.

---

## 1. Introduction to Reinforcement Learning

Reinforcement Learning (RL) is a subfield of machine learning concerned with how an intelligent **agent** learns to make decisions by interacting with an **environment** to maximise a cumulative numerical **reward** signal.

Unlike supervised learning, RL does not rely on labelled data. Instead, the agent:
1. Observes the current **state** *s* of the environment
2. Selects an **action** *a* from its action space
3. Receives a **reward** *r* and transitions to a new state *s'*
4. Updates its knowledge to improve future decisions

This trial-and-error learning framework has been applied successfully to game playing (AlphaGo, Atari), robotics, autonomous driving, and natural language processing.

**Key concepts:**
| Term | Definition |
|---|---|
| Agent | The learner / decision-maker |
| Environment | The world the agent interacts with |
| State (*s*) | A representation of the current situation |
| Action (*a*) | A choice the agent can make |
| Reward (*r*) | Scalar feedback signal from environment |
| Policy (*π*) | A mapping from states to actions |
| Value function | Expected cumulative reward from a state |

---

## 2. Markov Decision Process (MDP)

A Markov Decision Process is the formal mathematical framework used to model RL problems. An MDP is defined by the tuple **(S, A, P, R, γ)**:

| Symbol | Meaning |
|---|---|
| **S** | Finite set of states |
| **A** | Finite set of actions |
| **P(s'∣s,a)** | Transition probability: probability of reaching *s'* from *s* via *a* |
| **R(s,a)** | Expected reward for taking action *a* in state *s* |
| **γ ∈ [0,1)** | Discount factor: controls importance of future rewards |

**The Markov Property** states that the future state depends only on the current state and action, not on the history of past states:

```
P(s_{t+1} | s_t, a_t, s_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

**Return (Cumulative Reward):**
```
G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ... = Σ γᵏ · r_{t+k}
```

The agent's goal is to learn a policy *π* that maximises the expected return G_t.

**In our Grid World:**
- **S** = 36 states (6×6 grid, encoded as integer index `row × 6 + col`)
- **A** = {Up, Down, Left, Right} → 4 actions
- **Transitions** are partly stochastic due to the dynamic obstacle
- **γ = 0.99** (values future rewards nearly as much as immediate ones)

---

## 3. Q-Learning Theory

### 3.1 Action-Value Function

The **Q-function** Q(s, a) represents the expected cumulative reward of taking action *a* in state *s* and following the optimal policy thereafter:

```
Q*(s, a) = E[G_t | s_t = s, a_t = a, π*]
```

### 3.2 The Bellman Optimality Equation

The optimal Q-function satisfies:

```
Q*(s, a) = R(s, a) + γ · max_{a'} Q*(s', a')
```

### 3.3 Q-Learning Update Rule

Q-Learning is an **off-policy, model-free** algorithm that directly learns Q* using:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_{a'} Q(s', a') − Q(s, a) ]
                                └──────── TD Target ─────────┘
```

Where:
- **α** = learning rate (0.1) — how much to update Q on each step
- **γ** = discount factor (0.99) — importance of future rewards
- **r + γ · max Q(s', a')** = Temporal Difference (TD) target
- **r + γ · max Q(s', a') − Q(s, a)** = TD error

### 3.4 ε-Greedy Exploration

To balance exploration and exploitation:
- With probability **ε** → choose random action (explore)
- With probability **1−ε** → choose argmax Q(s, a) (exploit)
- ε decays from 1.0 → 0.01 using multiplicative decay (× 0.995 per episode)

### 3.5 Tabular Q-Table

For our 6×6 grid, the Q-table has shape **36 × 4** = 144 values, initialised to 0. Each cell Q[s][a] stores the estimated value of taking action *a* in state *s*.

---

## 4. Deep Q-Network (DQN) Theory

### 4.1 Motivation

Tabular Q-Learning fails when the state space is very large (e.g., image pixels, continuous values) — storing a Q-table becomes infeasible. DQN replaces the table with a **neural network** that approximates Q(s, a; θ) for all actions simultaneously.

### 4.2 Network Architecture

```
Input Layer    :  36 neurons  (one-hot encoded state)
Hidden Layer 1 :  64 neurons  (ReLU activation)
Hidden Layer 2 :  64 neurons  (ReLU activation)
Output Layer   :   4 neurons  (one Q-value per action)
```

### 4.3 Loss Function

DQN minimises the Mean Squared Error between predicted and target Q-values:

```
L(θ) = E[(y − Q(s, a; θ))²]

where  y = r + γ · max_{a'} Q(s', a'; θ⁻)
```

θ = online network weights | θ⁻ = target network weights (frozen)

### 4.4 Experience Replay

Instead of learning from consecutive correlated transitions, DQN stores past transitions **(s, a, r, s', done)** in a **Replay Buffer** (capacity = 10,000). At each step, a random **mini-batch** of 64 transitions is sampled, breaking temporal correlations and stabilising training.

### 4.5 Target Network

A separate **target network** (identical architecture) provides stable Q-value targets. Its weights **θ⁻** are frozen and copied from the online network every **100 steps**. This prevents the feedback loop instability that would arise from training against a rapidly-changing target.

### 4.6 Optimizer

**Adam optimizer** (lr = 0.001) is used for adaptive gradient-based weight updates.

---

## 5. Environment Design

### 5.1 Grid Layout

```
┌─────────────┐
│ A . . . . . │   Row 0
│ . X . . . . │   Row 1   (X = static obstacle at (1,1))
│ . . . . D . │   Row 2   (D = dynamic obstacle, starts at (2,4))
│ . . . X . . │   Row 3   (X = static obstacle at (3,3))
│ . . . . . . │   Row 4
│ . . . . . G │   Row 5   (G = goal at (5,5))
└─────────────┘
  0 1 2 3 4 5
```

### 5.2 State Representation

Each cell is encoded as a single integer:
```
state = row × 6 + col      (range: 0 to 35)
```
This integer is directly usable as a Q-table row index, and is one-hot encoded for DQN input.

### 5.3 Action Space

| Action | Code | Effect |
|---|---|---|
| Up | 0 | row − 1 |
| Down | 1 | row + 1 |
| Left | 2 | col − 1 |
| Right | 3 | col + 1 |

Wall collisions are handled by clamping — the agent stays in place.

### 5.4 Reward Structure

| Event | Reward | Episode Ends? |
|---|---|---|
| Reach goal (5,5) | +10 | Yes |
| Hit static obstacle | −10 | Yes |
| Hit dynamic obstacle | −10 | Yes |
| Normal step | −1 | No |
| Exceed 50 steps | −1 (last) | Yes |

### 5.5 Dynamic Obstacle

The dynamic obstacle moves randomly at every step to a valid adjacent cell (cannot enter start, goal, or static obstacle cells). This introduces **non-stationarity** — the optimal path changes each episode, making the problem harder and more realistic.

### 5.6 Terminal Conditions
1. Agent reaches the goal
2. Agent hits any obstacle (static or dynamic)
3. 50 steps elapsed without reaching goal

---

## 6. Experimental Results

### 6.1 Hyperparameters

| Parameter | Q-Learning | DQN |
|---|---|---|
| Episodes | 5,000 | 5,000 |
| Learning rate (α) | 0.1 | 0.001 (Adam) |
| Discount factor (γ) | 0.99 | 0.99 |
| ε initial | 1.0 | 1.0 |
| ε minimum | 0.01 | 0.01 |
| ε decay | 0.995 / episode | 0.995 / episode |
| Replay buffer | — | 10,000 |
| Batch size | — | 64 |
| Target update | — | Every 100 steps |
| Hidden layers | — | 64 → 64 |

### 6.2 Training Progression — Q-Learning

| Episode | Avg Reward (last 100) | ε |
|---|---|---|
| 500 | ≈ −5.0 | 0.08 |
| 1,000 | ≈ −1.0 | 0.01 |
| 2,000 | ≈ −0.3 | 0.01 |
| 5,000 | ≈ 0.0 | 0.01 |

### 6.3 Training Progression — DQN

| Episode | Avg Reward (last 100) | ε |
|---|---|---|
| 500 | ≈ −2.5 | 0.08 |
| 1,000 | ≈ −0.1 | 0.01 |
| 2,000 | ≈ −0.5 | 0.01 |
| 5,000 | ≈ 0.0 | 0.01 |

Both agents converge to an average reward near **0**, which is the theoretical optimum (minimum path length = 10 steps → 10 steps × (−1) + 10 = 0).

---

## 7. Comparative Analysis

### 7.1 Convergence Speed

Q-Learning converges faster because its tabular update is **exact** — each Q(s, a) value is updated directly with zero approximation error.  
DQN requires the network weights to stabilise over many gradient steps and for the replay buffer to accumulate sufficient diverse experience before meaningful learning begins.

### 7.2 Final Performance

Both agents achieve near-identical final average rewards. For a small discrete environment like a 6×6 grid, the tabular Q-table is perfectly expressive — DQN's neural approximation offers no representational advantage here.

### 7.3 Effect of the Dynamic Obstacle

The dynamic obstacle makes the environment **partially stochastic**. Neither agent can learn a single fixed optimal path because the dynamic obstacle may randomly occupy it at any episode. Both agents exhibit persistent negative reward spikes throughout training, even after convergence. DQN's experience replay partially smooths these effects by averaging over many past transitions; Q-Learning reacts more sharply to individual surprise collisions.

### 7.4 Stability

Q-Learning's rolling average is smoother post-convergence due to the deterministic, exact nature of the tabular update.  
DQN shows higher variance from:
- Mini-batch sampling noise
- Bootstrapping from a periodically-delayed target network
- Gradient step overshooting near convergence

### 7.5 Scalability

| Criterion | Q-Learning | DQN |
|---|---|---|
| Small state spaces (≤ 10⁴) | ✅ Ideal | ✅ Works |
| Large / continuous state spaces | ❌ Infeasible | ✅ Required |
| Training speed (CPU) | ✅ Fast | ⚠ Slower |
| Memory usage | ✅ Low | ⚠ Replay buffer |
| Generalisation to unseen states | ❌ None | ✅ Yes |

**Verdict:** Q-Learning is the better choice for this small grid. DQN becomes essential when state spaces grow large (e.g., image-based Atari games, robotics with continuous sensors).

---

## 8. Conclusion

This project successfully demonstrated the design and training of two reinforcement learning agents on a custom 6×6 Grid World environment. The environment incorporated a dynamic obstacle to model stochastic real-world conditions.

**Key findings:**
1. Both Q-Learning and DQN learned effective policies, converging to near-optimal average rewards of ≈ 0 within 5,000 episodes.
2. Q-Learning converged faster and was more stable due to its exact tabular updates.
3. DQN demonstrated the three signature techniques (experience replay, target network, neural function approximation) that make it scale to complex environments.
4. The dynamic obstacle demonstrated how stochasticity in environments prevents perfect policy convergence in finite training.
5. The ε-greedy decay strategy was critical in transitioning both agents from exploration to exploitation.

This project reinforces the trade-off between simplicity and scalability in RL algorithm design: start with the simplest algorithm that works, and resort to deep learning only when the problem demands it.

---

## 9. References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
3. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8(3–4), 279–292.
4. PyTorch Documentation: https://pytorch.org/docs/stable/
5. OpenAI Gym Documentation: https://www.gymlibrary.dev/

---

## 10. Viva Questions and Answers

---

**Q1. What is the Markov Property, and why is it important in RL?**

**A:** The Markov Property states that the future state depends only on the current state and action, not on the history of previous states: P(s'|s,a,s_{t-1},...) = P(s'|s,a). It is important because it allows the agent to make decisions using only the current state, drastically simplifying the problem. Without it, the agent would need to remember the entire history of interactions.

---

**Q2. What is the difference between on-policy and off-policy learning? Which is Q-Learning?**

**A:** In **on-policy** learning (e.g., SARSA), the agent learns the value of the policy it is currently following, including its exploratory actions. In **off-policy** learning (e.g., Q-Learning), the agent learns the optimal policy regardless of the exploratory actions it takes. Q-Learning is **off-policy** — it updates Q-values using the greedy action `max Q(s', a')` even if the agent actually took a random exploratory action.

---

**Q3. Why do we use ε-greedy exploration, and what happens if ε is always 0 or always 1?**

**A:** ε-greedy balances exploration (trying new actions to discover better strategies) and exploitation (using known good actions). If ε = 0 always (pure exploitation): the agent never explores and may get stuck in a local optimum. If ε = 1 always (pure exploration): the agent always acts randomly and never learns to use knowledge it has accumulated.

---

**Q4. What is the role of the discount factor γ? What happens at γ = 0 and γ = 1?**

**A:** γ controls how much the agent values future rewards relative to immediate rewards. The return is G_t = Σ γᵏ rₜ₊ₖ. At γ = 0: the agent is completely myopic — it only cares about the immediate reward. At γ = 1: all future rewards are equally important (valid only for episodic tasks with guaranteed termination). γ = 0.99 means the agent values future rewards nearly as much as immediate ones, encouraging long-term planning.

---

**Q5. Why does DQN use a replay buffer? What problem does it solve?**

**A:** DQN uses an experience replay buffer to store past transitions (s, a, r, s', done) and sample random mini-batches for training. It solves two problems: (1) **Correlation:** Consecutive transitions are highly correlated — training on them sequentially would cause the network to overfit to recent experience. Random sampling breaks this correlation. (2) **Sample efficiency:** Each transition can be reused multiple times, making learning more data-efficient.

---

**Q6. Why does DQN have two networks (online and target)? What happens without the target network?**

**A:** The target network provides stable Q-value targets: y = r + γ · max Q(s', a'; θ⁻). Without it, both the predicted and target values are computed from the same network θ. As θ is updated, both values shift simultaneously, creating a moving target — like trying to catch a moving carrot. This feedback loop causes instability and often divergence. The target network (updated every 100 steps) keeps the target fixed for short intervals, stabilising training.

---

**Q7. What is the Bellman equation, and how is it used to update Q-values?**

**A:** The Bellman Optimality Equation expresses the relationship between a state's Q-value and its successor states: Q*(s,a) = R(s,a) + γ · max_{a'} Q*(s',a'). In Q-Learning, we use this as an update target: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]. The term in brackets is the **TD error** — the difference between the estimated and target Q-values. The agent gradually reduces this error over many episodes until Q converges to Q*.

---

**Q8. Why is the state encoded as a one-hot vector for DQN but as an integer for Q-Learning?**

**A:** In Q-Learning, the integer state index directly accesses a row in the tabular Q-table (O(1) lookup). In DQN, the neural network needs a **continuous, real-valued input vector**. Using the raw integer (a single scalar) is problematic because the network would incorrectly interpret states close in integer value (e.g., 10 and 11) as similar in meaning. One-hot encoding gives each state a unique orthogonal vector representation, preventing the network from falsely inferring proximity between unrelated states.

---

**Q9. What is the effect of the dynamic obstacle on learning? How does each algorithm handle it?**

**A:** The dynamic obstacle introduces stochasticity — the environment's transition dynamics are not fully deterministic. Even the optimal policy cannot guarantee reaching the goal every episode because the obstacle can randomly walk into the agent's path. Q-Learning handles this by gradually averaging Q-values over many encounters; however, a single surprise collision causes a sharp one-step update. DQN's replay buffer naturally averages over many past transitions, giving it a softer, more distributed response to the random obstacle's behavior.

---

**Q10. When would you prefer DQN over Q-Learning in a real-world application?**

**A:** We prefer DQN over Q-Learning when: (1) The **state space is too large** to fit in a table (e.g., image pixels → millions of states in Atari). (2) The state is **continuous** (e.g., position and velocity in robotics). (3) We need **generalisation** — the neural network can interpolate Q-values for states not seen during training, which a table cannot do. (4) Input is **high-dimensional** (e.g., camera images, sensor arrays). For our 6×6 grid world (36 states), Q-Learning is the more practical choice — DQN's overhead is not justified by the problem's simplicity.

---

*End of Report*
