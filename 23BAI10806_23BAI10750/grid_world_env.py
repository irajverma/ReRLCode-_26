"""
grid_world_env.py
=================
Custom 6x6 Grid World Environment for Reinforcement Learning.

Features:
  - 6x6 grid with a single start and goal state
  - 2 static obstacles
  - 1 dynamic obstacle (moves randomly every step)
  - Reward: +10 (goal), -10 (obstacle hit), -1 (per step)
  - Episode ends on: goal reached | obstacle hit | 50 steps exceeded
  - State: single integer index  ->  row * GRID_SIZE + col
  - Actions: 0=Up, 1=Down, 2=Left, 3=Right
"""

import random


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_SIZE    = 6          # 6×6 grid
MAX_STEPS    = 50         # Maximum steps per episode

# Reward values
REWARD_GOAL      =  10
REWARD_OBSTACLE  = -10
REWARD_STEP      =  -1

# Action mappings: action -> (row_delta, col_delta)
ACTION_DELTAS = {
    0: (-1,  0),   # Up
    1: ( 1,  0),   # Down
    2: ( 0, -1),   # Left
    3: ( 0,  1),   # Right
}
ACTION_NAMES = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

# Grid cell symbols used in render()
SYMBOL_EMPTY   = "."
SYMBOL_AGENT   = "A"
SYMBOL_GOAL    = "G"
SYMBOL_STATIC  = "X"   # Static obstacle
SYMBOL_DYNAMIC = "D"   # Dynamic obstacle


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _pos_to_state(row: int, col: int) -> int:
    """Convert (row, col) to a single integer state index."""
    return row * GRID_SIZE + col


def _state_to_pos(state: int):
    """Convert a single integer state index back to (row, col)."""
    return divmod(state, GRID_SIZE)


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp value to [lo, hi] (inclusive)."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# GridWorldEnv
# ---------------------------------------------------------------------------
class GridWorldEnv:
    """
    6×6 custom Grid World environment compatible with Q-learning / DQN agents.

    Grid layout (default):
        Start  : (0, 0)  — top-left corner
        Goal   : (5, 5)  — bottom-right corner
        Static obstacles : (1, 1) and (3, 3)
        Dynamic obstacle : starts at (2, 4), moves randomly each step
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self):
        # Fixed positions
        self.start_pos          = (0, 0)
        self.goal_pos           = (5, 5)
        self.static_obstacles   = [(1, 1), (3, 3)]   # Never move
        self.dynamic_start_pos  = (2, 4)              # Resets here each episode

        # Derived constants
        self.n_states  = GRID_SIZE * GRID_SIZE        # 36 discrete states
        self.n_actions = len(ACTION_DELTAS)            # 4 actions

        # Episode state (populated on reset)
        self.agent_pos   = None
        self.dynamic_pos = None
        self.step_count  = 0
        self.done        = False

    # ------------------------------------------------------------------
    # reset()  →  initial state (int)
    # ------------------------------------------------------------------
    def reset(self) -> int:
        """
        Reset the environment to its initial configuration.

        Returns
        -------
        int
            The starting state as a single integer index.
        """
        self.agent_pos   = list(self.start_pos)        # Mutable [row, col]
        self.dynamic_pos = list(self.dynamic_start_pos)
        self.step_count  = 0
        self.done        = False
        return _pos_to_state(*self.agent_pos)

    # ------------------------------------------------------------------
    # step(action)  →  (next_state, reward, done)
    # ------------------------------------------------------------------
    def step(self, action: int):
        """
        Execute one environment step.

        Parameters
        ----------
        action : int
            0=Up | 1=Down | 2=Left | 3=Right

        Returns
        -------
        next_state : int
            New state as a single integer index.
        reward : float
            Reward received this step.
        done : bool
            True if the episode has ended.
        """
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() before stepping.")

        # ---- 1. Move agent -----------------------------------------------
        dr, dc = ACTION_DELTAS[action]
        new_row = _clamp(self.agent_pos[0] + dr, 0, GRID_SIZE - 1)
        new_col = _clamp(self.agent_pos[1] + dc, 0, GRID_SIZE - 1)
        self.agent_pos = [new_row, new_col]

        # ---- 2. Move dynamic obstacle (random valid step) ----------------
        self._move_dynamic_obstacle()

        # ---- 3. Increment step counter -----------------------------------
        self.step_count += 1

        # ---- 4. Compute reward and check termination ----------------------
        reward, self.done = self._evaluate()

        next_state = _pos_to_state(*self.agent_pos)
        return next_state, reward, self.done

    # ------------------------------------------------------------------
    # render()
    # ------------------------------------------------------------------
    def render(self):
        """
        Print a human-readable ASCII representation of the current grid.

        Legend
        ------
        A  : Agent
        G  : Goal
        X  : Static obstacle
        D  : Dynamic obstacle
        .  : Empty cell
        """
        agent   = tuple(self.agent_pos)
        dynamic = tuple(self.dynamic_pos)

        print(f"\n┌{'─' * (GRID_SIZE * 2 + 1)}┐")
        for row in range(GRID_SIZE):
            row_str = "│ "
            for col in range(GRID_SIZE):
                pos = (row, col)
                if pos == agent:
                    cell = SYMBOL_AGENT
                elif pos == self.goal_pos:
                    cell = SYMBOL_GOAL
                elif pos == dynamic:
                    cell = SYMBOL_DYNAMIC
                elif pos in self.static_obstacles:
                    cell = SYMBOL_STATIC
                else:
                    cell = SYMBOL_EMPTY
                row_str += cell + " "
            row_str += "│"
            print(row_str)
        print(f"└{'─' * (GRID_SIZE * 2 + 1)}┘")
        print(f"  Step: {self.step_count} | "
              f"Agent: {tuple(self.agent_pos)} | "
              f"State: {_pos_to_state(*self.agent_pos)}")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def state_to_pos(self, state: int):
        """Convert integer state index → (row, col). Useful for agents."""
        return _state_to_pos(state)

    def pos_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) → integer state index. Useful for agents."""
        return _pos_to_state(row, col)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _move_dynamic_obstacle(self):
        """
        Move the dynamic obstacle one step in a random valid direction.
        A direction is valid if it keeps the obstacle inside the grid
        and does NOT land on the start, goal, or a static obstacle.
        """
        forbidden = {self.start_pos, self.goal_pos} | set(self.static_obstacles)

        candidates = []
        for dr, dc in ACTION_DELTAS.values():
            nr = _clamp(self.dynamic_pos[0] + dr, 0, GRID_SIZE - 1)
            nc = _clamp(self.dynamic_pos[1] + dc, 0, GRID_SIZE - 1)
            if (nr, nc) not in forbidden:
                candidates.append((nr, nc))

        if candidates:
            self.dynamic_pos = list(random.choice(candidates))

    def _evaluate(self):
        """
        Determine the reward and whether the episode is over.

        Returns
        -------
        (reward : float, done : bool)
        """
        pos = tuple(self.agent_pos)

        # --- Goal reached -------------------------------------------------
        if pos == self.goal_pos:
            return REWARD_GOAL, True

        # --- Hit a static obstacle ----------------------------------------
        if pos in self.static_obstacles:
            return REWARD_OBSTACLE, True

        # --- Hit the dynamic obstacle --------------------------------------
        if pos == tuple(self.dynamic_pos):
            return REWARD_OBSTACLE, True

        # --- Step limit exceeded ------------------------------------------
        if self.step_count >= MAX_STEPS:
            return REWARD_STEP, True        # timeout; still apply step penalty

        # --- Normal step --------------------------------------------------
        return REWARD_STEP, False


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = GridWorldEnv()
    state = env.reset()
    print(f"Environment created  |  n_states={env.n_states}  |  n_actions={env.n_actions}")
    print(f"Initial state index: {state}")
    env.render()

    # Run a short random episode to verify step() and render()
    total_reward = 0
    for _ in range(10):
        action = random.randint(0, env.n_actions - 1)
        next_state, reward, done = env.step(action)
        total_reward += reward
        print(f"\nAction: {ACTION_NAMES[action]:5s}  |  "
              f"Next state: {next_state:2d}  |  "
              f"Reward: {reward:+.0f}  |  Done: {done}")
        env.render()
        if done:
            print(f"\nEpisode ended after {env.step_count} steps. "
                  f"Total reward: {total_reward}")
            break
