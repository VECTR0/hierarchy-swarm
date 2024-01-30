from abc import ABC, abstractmethod

class QBrain(ABC):
    def __init__(self, id):
        self.id = id
    
    def act(self, observation, environment):
        return environment.action_space(self.id).sample()