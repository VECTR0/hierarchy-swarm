from mods.agents.agent import Agent

class HardcodedAgent(Agent):
    def __init__(self, id):
        self.id = id
    
    def act(self, observation, environment):
        return environment.action_space(self.id).sample()