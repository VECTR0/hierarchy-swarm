from datetime import datetime as dt

class Agent:
    def __init__(self, environment):
        self.environment = environment
        self.sensors = dict()
        self.actions = dict()
        self.think_interval = 10
        self.think_next_time = environment.time_now + self.think_interval
        self.brain = None

    def set_brain(self, brain):
        self.brain = brain

    def think(self):
        pass

    def add_sensor(self, name, sensor):
        self.sensors[name] = sensor
    
    def add_action(self, name, action):
        self.actions[name] = action

    def tick(self):
        if self.environment.time_now > self.think_next_time:
            self.think()
            self.think_next_time = self.environment.time_now + self.think_interval