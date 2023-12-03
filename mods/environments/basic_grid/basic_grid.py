from collections import namedtuple
from copy import copy
import functools
import random
from sys import flags
from PIL import Image
from gymnasium.spaces import Discrete, Box, Tuple, Dict
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, MultiDiscrete
from zmq import has
import pygame

from mods.environments.basic_grid.basic_agent import BasicAgent

'''
GRID:
(X,Y, W,F,D)
X: width
Y: height
W: walkable
F: food
D: danger
'''
DANGER = 0
FOOD = 1
WALK = 2


IDLE = 0
MOVE_UP = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
MOVE_RIGHT = 4
EAT = 5
EMIT_SIGNAL = 6



class BasicGrid(ParallelEnv):
    metadata = {
        "name": "basic_grid",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 65
    }

    def __init__(self, agents_num=10, grid_size=(10, 10), render_mode=None, max_iterations=1000, grid_image=None):
        self.agents_num = agents_num
        self.grid_size = grid_size
        self.render_mode = render_mode
        self.max_iterations = max_iterations
        self.iteration = 0
        self.possible_agents = [BasicAgent((0, 0), "a_" + str(r)) for r in range(agents_num)]
        self.agent_id_mapping = dict(zip([a.id for a in self.possible_agents], self.possible_agents))


        self.grid = np.full((*self.grid_size,3), 0, dtype=np.uint8)
        self._init_basic_grid()
        if grid_image is not None:
            self._load_grid_from_file(grid_image)

        self._occupied_positions = set()
        self.emitted_signals = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.window_size = 500
        self.window = None
        self.clock = None

        self.attributes = namedtuple("attributes", ["danger_energy_drain", "food_energy_gain"])
        self.attributes.danger_energy_drain = 1 # type: ignore
        self.attributes.food_energy_gain = 1 # type: ignore

    def _load_grid_from_file(self, image):
        self.grid_size = image.size
        image = image.load()
        self.grid = np.full((*self.grid_size,3), 0, dtype=np.uint8)
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                r = image[x,y][0]
                g = image[x,y][1]
                b = image[x,y][2]
                self.grid[x, y, DANGER] = r
                self.grid[x, y, FOOD] = g
                self.grid[x, y, WALK] = b

    def _init_basic_grid(self):
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):                
                self.grid[x, y] = np.random.randint(0, 255, size=3, dtype=np.uint8)
                if np.linalg.norm(np.array([x,y]) - np.array([self.grid_size[0]/2, self.grid_size[1]/2])) > 20:
                    self.grid[x, y,0] = 127
                else:
                    self.grid[x, y,0] = 255

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        # super().reset(seed=seed) TODO: check if this is needed
        self.agents = copy(self.possible_agents)
        self._occupied_positions = set()
        self.emitted_signals = []
        # places agents on random positions which are walkable
        for agent in self.agents:
            possible_positions = []
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    if self.grid[x, y][WALK] > 255-100:
                        if (x, y) not in self._occupied_positions:
                            possible_positions.append((x, y))
            assert len(possible_positions) > 0
            agent.position = possible_positions[np.random.randint(0, len(possible_positions))]
            self._occupied_positions.add(agent.position)
            agent.attributes.energy = 100

        self.iteration = 0
        observations = {a.id: {} for a in self.agents}
        infos = {a.id: {} for a in self.agents}
        
        if self.render_mode == "human":
            self._render_frame()
        return observations, infos

    def step(self, actions):
        self.emitted_signals = []
        for agent in self.agents:
            action = actions[agent.id]
            discrete, box = action
            if discrete == MOVE_DOWN:
                if agent.position[1] < self.grid_size[1]-1:
                    new_position = (agent.position[0], agent.position[1]+1) 
                    if np.random.randint(0, 255, size=1, dtype=np.uint8) < self.grid[new_position][WALK]: #
                        if new_position not in self._occupied_positions:
                            self._occupied_positions.remove(agent.position)
                            agent.position = new_position
                            self._occupied_positions.add(agent.position)
                            agent.attributes.energy -= agent.attributes.movement_energy_cost # type: ignore
            elif discrete == MOVE_UP:
                if agent.position[1] > 0:
                    new_position = (agent.position[0], agent.position[1]-1) 
                    if np.random.randint(0, 255, size=1, dtype=np.uint8) < self.grid[new_position][WALK]:
                        if new_position not in self._occupied_positions:
                            self._occupied_positions.remove(agent.position)
                            agent.position = new_position
                            self._occupied_positions.add(agent.position)
                            agent.attributes.energy -= agent.attributes.movement_energy_cost # type: ignore
            elif discrete == MOVE_LEFT:
                if agent.position[0] > 0:
                    new_position = (agent.position[0]-1, agent.position[1]) 
                    if np.random.randint(0, 255, size=1, dtype=np.uint8) < self.grid[new_position][WALK]:
                        if new_position not in self._occupied_positions:
                            self._occupied_positions.remove(agent.position)
                            agent.position = new_position
                            self._occupied_positions.add(agent.position)
                            agent.attributes.energy -= agent.attributes.movement_energy_cost # type: ignore
            elif discrete == MOVE_RIGHT:
                if agent.position[0] < self.grid_size[0]-1:
                    new_position = (agent.position[0]+1, agent.position[1]) 
                    if np.random.randint(0, 255, size=1, dtype=np.uint8) < self.grid[new_position][WALK]:
                        if new_position not in self._occupied_positions:
                            self._occupied_positions.remove(agent.position)
                            agent.position = new_position
                            self._occupied_positions.add(agent.position)
                            agent.attributes.energy -= agent.attributes.movement_energy_cost # type: ignore
                    
            elif discrete == EAT:
                if self.grid[agent.position][FOOD] > 40:
                    agent.attributes.energy += 40
                    self.grid[agent.position][FOOD] -= 40
            
            elif discrete == EMIT_SIGNAL:
                range = box[0] * 10
                self.emitted_signals.append({"position": agent.position, "range": range, "type": box[1]})
                agent.attributes.energy -= agent.attributes.signal_emit_cost * box[0] # type: ignore

            agent.attributes.energy -= self.grid[agent.position][DANGER] * self.attributes.danger_energy_drain # type: ignore
            agent.attributes.energy -= agent.attributes.idle_energy_cost
        # check for terminations
        terminations = {a.id: a.attributes.energy <= 0 for a in self.agents}
        # picks random agent and sets its termination to True
        # terminations[np.random.choice(self.agents)] = True

        rewards = {a.id: 0 for a in self.agents}
        for agent in self.agents:
            if agent.attributes.energy > agent.attributes.max_energy/2:
                rewards[agent.id] += 1
            if terminations[agent.id]:
                rewards[agent.id] -= 10
        # check for truncations
        truncations = {a.id: False for a in self.agents}
        self.iteration += 1
        if self.iteration >= self.max_iterations:
            truncations = {a.id: True for a in self.agents}
            for agent in self.agents:
                rewards[agent.id] += 10

        observations = {a.id: {} for a in self.agents}
        # observations = 
        infos = {a.id: {} for a in self.agents}
        if all(terminations.values()) or all(truncations.values()):
            self.agents = []
        
        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminations, truncations, infos
    
    def _observation(self, agent):
        pass

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            flags = pygame.DOUBLEBUF | pygame.HWSURFACE
            self.window = pygame.display.set_mode((self.window_size, self.window_size), flags, 24)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        canvas.fill((127, 127, 127))
        pixel_size = (self.window_size / max(self.grid_size))
        
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                
                danger, food, walk  = self.grid[x, y]
                
                walk /= 255.0
                food /= 255.0
                danger /= 255.0
                def lerp(a, b, t):
                    return a + (b - a) * t

                def lerp_tuple(a, b, t):
                    return tuple(lerp(a[i], b[i], t) for i in range(len(a)))

                base = (1,1,1)
                color = base
                if danger != 0 or food != 0:
                    blended_alpha = danger + food*(1-danger)
                    
                    red = (1*danger) / blended_alpha
                    green = (1*food*(1-danger)) / blended_alpha
                    blue = 0

                    color = lerp_tuple(base, (red, green, blue), blended_alpha)
                color = tuple([int(c * 255.0 ) for c in color])
                color = lerp_tuple((0,0,0), color, walk)
              
                assert len(color) == 3
                
                pygame.draw.rect(
                    canvas,
                    color,
                    (
                        x * pixel_size,
                        y * pixel_size,
                        pixel_size,
                        pixel_size,
                    ),
                ) 

        for signal in self.emitted_signals:
            color = (255-255*signal["type"], 255*signal["type"], 0, 100)
            pygame.draw.circle(
                canvas,
                color,
                (signal["position"][0] * pixel_size+pixel_size/2, signal["position"][1] * pixel_size+pixel_size/2),
                signal["range"] * pixel_size / 2
            )

        for agent in self.agents:
            pygame.draw.circle(
                canvas,
                (255,255,255),
                (agent.position[0] * pixel_size+pixel_size/2, agent.position[1] * pixel_size+pixel_size/2),
                pixel_size / 3,
            )
            pygame.draw.circle(
                canvas,
                (0,0,0),
                (agent.position[0] * pixel_size+pixel_size/2, agent.position[1] * pixel_size+pixel_size/2),
                pixel_size / 3,
                width=1
            )

        # draws occupied position with /clockrender
        # for position in self._occupied_positions:
        #     pygame.draw.lines(
        #         canvas,
        #         (0, 0, 0),
        #         True,
        #         (
        #             (position[0] * pixel_size, position[1] * pixel_size),
        #             (position[0] * pixel_size + pixel_size, position[1] * pixel_size + pixel_size),
        #         ),
        #         width=2,
        #     )

        for x in range(max(self.grid_size) + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pixel_size * x),
                (self.window_size, pixel_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                0,
                (pixel_size * x, 0),
                (pixel_size * x, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            assert self.window is not None
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            assert self.clock is not None
            self.clock.tick(self.metadata["render_fps"])
        else: 
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    '''
    ENERGY
    SINGAL_AVERAGE
    BOXES_AROUND
    '''

    # @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        if hasattr(self, 'observation_spaces') is False:
            self.observation_spaces = dict()
        if agent_id not in self.observation_spaces or self.observation_spaces[agent_id] is None:
            self.observation_spaces[agent_id] = Dict({
                'grid_perception': MultiDiscrete([3] * 5),
                'food_content': Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'signals_from_other_agents': Box(low=0, high=1, shape=(1,), dtype=np.float32)
            })
        return self.observation_spaces[agent_id]

    '''
    DISCRETE ACTIONS:
        IDLE
        MOVE_UP
        MOVE_DOWN
        MOVE_LEFT
        MOVE_RIGHT
        EAT
    BOX ACTIONS:
        SIGNAL: STRENGTH TYPE
    '''

    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id):
        if hasattr(self, 'action_spaces') is False:
            self.action_spaces = dict()
        if agent_id not in self.action_spaces or self.action_spaces[agent_id] is None:
            self.action_spaces[agent_id] = Tuple((Discrete(7,), Box(low=0, high=1, shape=(2,), dtype=np.float32)))
        return self.action_spaces[agent_id]
