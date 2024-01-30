from mods.brains.brain import Brain

class RandomBrain(Brain):
    def __init__(self, id = None):
        self.id = id
    
    def act(self, observation, environment):
        return environment.action_space(self.id).sample()
    
    def clone(self):
        return RandomBrain(id=self.id)
    
    def reset(self):
        pass

    def serialize(self):
        return {
            'id': self.id
        }
    
    @staticmethod
    def deserialize(data):
        return RandomBrain(id=data['id'])
    
    def mutate(self, other):
        pass