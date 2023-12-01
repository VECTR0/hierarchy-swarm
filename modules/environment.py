class Environment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.time_now = 0
        self.agents = []
    
    def add_agents(self, agents):
        for agent in agents:
            agent.environment = self
            self.agents.append(agent)

    def simulate(self, iterations=1):
        for _ in range(iterations):
            self.tick()
            for agent in self.agents:
                agent.tick()

    def tick(self):
        self.time_now += 1

    def get_cell(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        else:
            return None

    def set_cell(self, x, y, value):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = value

    def print_grid(self):
        for row in self.grid:
            print(row)
