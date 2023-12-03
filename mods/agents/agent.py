from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def act(self, observation, environment):
        pass
