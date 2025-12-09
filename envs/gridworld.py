import numpy as np
from typing import Tuple, Dict

class GridWorldEnv:
    """
    Simple GridWorld:

    - 4x4 grid (states 0..15)
    - Start at top-left (0)
    - Goal at bottom-right (15)
    - Stepping into goal gives +1 reward and ends episode
    - Every other step gives -0.01 reward to encourage shorter paths
    """

    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4  # 0: up, 1: right, 2: down, 3: left

        self.start_state = 0
        self.goal_state = self.n_states - 1

        self.state = self.start_state

    def reset(self) -> int:
        """Reset to start state, return initial state."""
        self.state = self.start_state
        return self.state

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        row = state // self.grid_size
        col = state % self.grid_size
        return row, col

    def _pos_to_state(self, row: int, col: int) -> int:
        return row * self.grid_size + col

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Take an action in the environment.

        Returns:
            next_state, reward, done, info
        """
        assert 0 <= action < self.n_actions, "Invalid action"

        row, col = self._state_to_pos(self.state)

        if action == 0:   # up
            row = max(row - 1, 0)
        elif action == 1: # right
            col = min(col + 1, self.grid_size - 1)
        elif action == 2: # down
            row = min(row + 1, self.grid_size - 1)
        elif action == 3: # left
            col = max(col - 1, 0)

        next_state = self._pos_to_state(row, col)
        self.state = next_state

        done = next_state == self.goal_state
        reward = 1.0 if done else -0.01

        return next_state, reward, done, {}
