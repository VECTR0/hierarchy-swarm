from abc import ABC, abstractmethod

class Brain(ABC):
    def __init__(self, id):
        self.id = id
        
    @abstractmethod
    def act(self, observation, environment):
        pass

    @abstractmethod
    def clone(self) -> 'Brain':
        pass
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def serialize(self):
        pass

    @staticmethod
    @abstractmethod
    def deserialize(data):
        pass

    @abstractmethod
    def mutate(self, other):
        pass