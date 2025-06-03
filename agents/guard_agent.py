import random
import pickle
from agents.base_agent import BaseAgent

class GuardAgent(BaseAgent):
    """
    Tabular Q-learning agent for the Guard in HeistEnv.
    Uses epsilon-greedy action selection and maintains a Q-table.
    """
    def __init__(self, action_space, alpha=0.1, gamma=0.99, epsilon=0.1):
        super().__init__(action_space)
        # Hyperparameters
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        # Q-table stored as a dict: {state: [Q(a0), Q(a1), ...]}
        self.q_table = {}

    def _ensure_state(self, state):
        """
        Ensure the state is in the Q-table.
        """
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(len(self.action_space))]

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        self._ensure_state(state)
        # Explore
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        # Exploit
        q_values = self.q_table[state]
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning update for a single transition.
        """
        self._ensure_state(state)
        self._ensure_state(next_state)
        q_current = self.q_table[state][action]
        q_next_max = max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * q_next_max
        td_delta = td_target - q_current
        self.q_table[state][action] += self.alpha * td_delta

    def save_q_table(self, filepath):
        """
        Save only the Q-table to a file.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filepath):
        """
        Load Q-table from file.
        """
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
