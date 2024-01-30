from mods.brains.brain import Brain

class HardcodedBrain(Brain):
    def __init__(self, id):
        self.id = id
    
    def act(self, observation, environment):
        return environment.action_space(self.id).sample()
    