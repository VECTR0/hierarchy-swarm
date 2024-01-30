from mods.brains.brain import Brain

class NEAT_Brain(Brain):
    def __init__(self, id):
        self.id = id
        
    def act(self, observation, environment):
        pass

    def clone(self):
        pass
    
    def reset(self):
        pass

    def serialize(self):
        pass

    @staticmethod
    def deserialize(data):
        pass

    def mutate(self, other):
        pass