import os
import pickle
from agents.base_agent import BaseAgent

class AgentBundle:
    def __init__(self, agent: BaseAgent, metadata: dict = None):
        if not isinstance(agent, BaseAgent):
            raise TypeError("agent must be an instance of BaseAgent")
        self.agent = agent
        self.metadata = metadata or {}

    def save(self, filepath: str):
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        bundle = {
            'metadata': self.metadata,
            'agent_state': self.agent,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            bundle = pickle.load(f)
        agent = bundle.get('agent_state')
        metadata = bundle.get('metadata')
        return cls(agent, metadata)

    def get_agent(self) -> BaseAgent:
        return self.agent

    def get_metadata(self) -> dict:
        return self.metadata
