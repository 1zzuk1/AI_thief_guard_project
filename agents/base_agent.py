import os
import abc
import pickle

class BaseAgent(abc.ABC):
    def __init__(self, action_space):
        self.action_space = action_space
        self.knowledge = {}

    @abc.abstractmethod
    def select_action(self, state):
        pass

    @abc.abstractmethod
    def update(self, state, action, reward, next_state, done):
        pass

    def save(self, filepath):
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            agent = pickle.load(f)
        if not isinstance(agent, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}")
        return agent

    def reset(self):
        self.knowledge.clear()
