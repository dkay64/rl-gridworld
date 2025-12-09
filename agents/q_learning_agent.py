import numpy as np

class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate

        # Q-table initialized to zeros
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state: int) -> int:
        # epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        best_next_action = int(np.argmax(self.Q[next_state]))
        td_target = reward + (0 if done else self.gamma * self.Q[next_state, best_next_action])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error
