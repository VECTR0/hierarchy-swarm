from copy import copy
import dis
from math import dist, sqrt
from unittest import signals
from gymnasium.spaces import Discrete, Box, Tuple, Dict
from networkx import adamic_adar_index
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, MultiDiscrete
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
DROP_FOOD = 7

class BasicGrid(ParallelEnv):
    metadata = {
        "name": "basic_grid",
        "render_modes": ["human", "rgb_array", None],
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

        self.attributes = {}
        self.attributes['danger_energy_drain'] = 1 
        self.attributes['food_energy_gain'] = 1 
        self.attributes['signal_max_range'] = 300 

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
                # if np.linalg.norm(np.array([x,y]) - np.array([self.grid_size[0]/2, self.grid_size[1]/2])) > 20:
                #     self.grid[x, y,0] = 127
                # else:
                #     self.grid[x, y,0] = 255

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
            agent.attributes.energy = agent.attributes.max_energy

        self.iteration = 0
        observations = {a.id: {} for a in self.agents}
        for agent in self.agents:
            observations[agent.id]['grid_perception'] = [[0] * 3] * 5
            observations[agent.id]['grid_perception'][0][DANGER] = self.grid[agent.position][DANGER]
            observations[agent.id]['grid_perception'][0][WALK] = self.grid[agent.position][WALK]
            observations[agent.id]['grid_perception'][0][FOOD] = self.grid[agent.position][FOOD]
            observations[agent.id]['grid_perception'][1][DANGER] = self.grid[agent.position[0], agent.position[1]-1][DANGER]/ 255.0 if agent.position[1] > 0 else 0
            observations[agent.id]['grid_perception'][1][WALK] = self.grid[agent.position[0], agent.position[1]-1][WALK]/ 255.0 if agent.position[1] > 0 else 0
            observations[agent.id]['grid_perception'][1][FOOD] = self.grid[agent.position[0], agent.position[1]-1][FOOD]/ 255.0 if agent.position[1] > 0 else 0
            observations[agent.id]['grid_perception'][2][DANGER] = self.grid[agent.position[0]+1, agent.position[1]][DANGER]/ 255.0 if agent.position[0] < self.grid_size[0]-1 else 0
            observations[agent.id]['grid_perception'][2][WALK] = self.grid[agent.position[0]+1, agent.position[1]][WALK]/ 255.0 if agent.position[0] < self.grid_size[0]-1 else 0
            observations[agent.id]['grid_perception'][2][FOOD] = self.grid[agent.position[0]+1, agent.position[1]][FOOD]/ 255.0 if agent.position[0] < self.grid_size[0]-1 else 0
            observations[agent.id]['grid_perception'][3][DANGER] = self.grid[agent.position[0], agent.position[1]+1][DANGER]/ 255.0 if agent.position[1] < self.grid_size[1]-1 else 0
            observations[agent.id]['grid_perception'][3][WALK] = self.grid[agent.position[0], agent.position[1]+1][WALK]/ 255.0 if agent.position[1] < self.grid_size[1]-1 else 0
            observations[agent.id]['grid_perception'][3][FOOD] = self.grid[agent.position[0], agent.position[1]+1][FOOD]/ 255.0 if agent.position[1] < self.grid_size[1]-1 else 0
            observations[agent.id]['grid_perception'][4][DANGER] = self.grid[agent.position[0]-1, agent.position[1]][DANGER]/ 255.0 if agent.position[0] > 0 else 0
            observations[agent.id]['grid_perception'][4][WALK] = self.grid[agent.position[0]-1, agent.position[1]][WALK]/ 255.0 if agent.position[0] > 0 else 0
            observations[agent.id]['grid_perception'][4][FOOD] = self.grid[agent.position[0]-1, agent.position[1]][FOOD]/ 255.0 if agent.position[0] > 0 else 0
            observations[agent.id]['average_signal_strength'] = 0 
            observations[agent.id]['average_signal_type'] = 0
            observations[agent.id]['average_signal_direction_x'] = 0
            observations[agent.id]['average_signal_direction_y'] = 0
            observations[agent.id]['closest_signal_strength'] = 0
            observations[agent.id]['closest_signal_type'] = 0
            observations[agent.id]['closest_signal_direction_x'] = 0
            observations[agent.id]['closest_signal_direction_y'] = 0
            observations[agent.id]['energy'] = agent.attributes.energy / agent.attributes.max_energy

        infos = {a.id: {} for a in self.agents}
        
        if self.render_mode == "human":
            self._render_frame()
        return observations, infos

    def step(self, actions):
        self.emitted_signals = []
        for agent in self.agents:
            action = actions[agent.id]
            discrete, box = action
            if agent.attributes.energy > 0:
                if discrete == MOVE_DOWN:
                    if agent.position[1] < self.grid_size[1]-1:
                        new_position = (agent.position[0], agent.position[1]+1) 
                        if np.random.randint(0, 255, size=1, dtype=np.uint8) < self.grid[new_position][WALK]: #
                            if new_position not in self._occupied_positions:
                                self._occupied_positions.remove(agent.position)
                                agent.position = new_position
                                self._occupied_positions.add(agent.position)
                                agent.attributes.energy -= agent.attributes.movement_energy_cost
                elif discrete == MOVE_UP:
                    if agent.position[1] > 0:
                        new_position = (agent.position[0], agent.position[1]-1) 
                        if np.random.randint(0, 255, size=1, dtype=np.uint8) < self.grid[new_position][WALK]:
                            if new_position not in self._occupied_positions:
                                self._occupied_positions.remove(agent.position)
                                agent.position = new_position
                                self._occupied_positions.add(agent.position)
                                agent.attributes.energy -= agent.attributes.movement_energy_cost
                elif discrete == MOVE_LEFT:
                    if agent.position[0] > 0:
                        new_position = (agent.position[0]-1, agent.position[1]) 
                        if np.random.randint(0, 255, size=1, dtype=np.uint8) < self.grid[new_position][WALK]:
                            if new_position not in self._occupied_positions:
                                self._occupied_positions.remove(agent.position)
                                agent.position = new_position
                                self._occupied_positions.add(agent.position)
                                agent.attributes.energy -= agent.attributes.movement_energy_cost 
                elif discrete == MOVE_RIGHT:
                    if agent.position[0] < self.grid_size[0]-1:
                        new_position = (agent.position[0]+1, agent.position[1]) 
                        if np.random.randint(0, 255, size=1, dtype=np.uint8) < self.grid[new_position][WALK]:
                            if new_position not in self._occupied_positions:
                                self._occupied_positions.remove(agent.position)
                                agent.position = new_position
                                self._occupied_positions.add(agent.position)
                                agent.attributes.energy -= agent.attributes.movement_energy_cost 
                        
                elif discrete == EAT:
                    EAT_AMOUNT = self.attributes['food_energy_gain']
                    food_amount = self.grid[agent.position][FOOD] / 255.0
                    food_amount = min(food_amount, EAT_AMOUNT, agent.attributes.max_energy - agent.attributes.energy)
                    agent.attributes.energy += food_amount
                    self.grid[agent.position][FOOD] -= int(food_amount * 255)

                elif discrete == DROP_FOOD:
                    DROP_AMOUNT = 0.1
                    if self.grid[agent.position][FOOD] > DROP_AMOUNT:
                        agent.attributes.energy -= DROP_AMOUNT
                        self.grid[agent.position][FOOD] += int(DROP_AMOUNT * 255)
                
                elif discrete == EMIT_SIGNAL:
                    range = box[0] * agent.attributes.signal_emit_range * self.attributes['signal_max_range']
                    if range > 0:
                        self.emitted_signals.append({"position": agent.position, "range": range, "type": box[1]})
                        agent.attributes.energy -= agent.attributes.signal_emit_cost * box[0] 
                
                agent.attributes.energy -= self.grid[agent.position][DANGER] / 255.0 * self.attributes['danger_energy_drain']
                agent.attributes.energy -= agent.attributes.idle_energy_cost
        
        observations = {a.id: {} for a in self.agents}
        received_signals = 0

        for agent in self.agents:
            avg_signal_stregth = 0
            avg_signal_type = 0
            avg_signal_direction_x = 0
            avg_signal_direction_y = 0
            avg_signal_count = 0
            avg_signal_weight_total = 0

            strongest_signal_strength = 0
            strongest_signal_type = 0
            strongest_signal_direction_x = 0
            strongest_signal_direction_y = 0
            
            for signal in self.emitted_signals:
                signal_direction_x = signal["position"][0] - agent.position[0]
                signal_direction_y = signal["position"][1] - agent.position[1]
                distane_to_signal = sqrt(signal_direction_x**2 + signal_direction_y**2)
                
                if distane_to_signal == 0:
                    continue
                signal_direction_x /= distane_to_signal
                signal_direction_y /= distane_to_signal

                signal_strength = 1 - distane_to_signal / signal["range"]
                signal_type = signal["type"]
                
                if signal_strength > agent.attributes.signal_hear_range:  # > < bug?
                    received_signals += 1
                    avg_signal_stregth += signal_strength
                    avg_signal_type += signal_type * signal_strength
                    avg_signal_direction_x += signal_direction_x * signal_strength
                    avg_signal_direction_y += signal_direction_y * signal_strength
                    avg_signal_count += 1
                    avg_signal_weight_total += signal_strength

                    if signal_strength > strongest_signal_strength:
                        strongest_signal_strength = signal_strength
                        strongest_signal_type = signal_type
                        strongest_signal_direction_x = signal_direction_x
                        strongest_signal_direction_y = signal_direction_y
            if avg_signal_count > 0:
                avg_signal_stregth /= avg_signal_weight_total 
                avg_signal_type /= avg_signal_weight_total
                avg_signal_direction_x /= avg_signal_weight_total
                avg_signal_direction_y /= avg_signal_weight_total
            
            observations[agent.id]['grid_perception'] = [[0] * 3] * 5
            observations[agent.id]['grid_perception'][0][DANGER] = self.grid[agent.position][DANGER]
            observations[agent.id]['grid_perception'][0][WALK] = self.grid[agent.position][WALK]
            observations[agent.id]['grid_perception'][0][FOOD] = self.grid[agent.position][FOOD]
            observations[agent.id]['grid_perception'][1][DANGER] = self.grid[agent.position[0], agent.position[1]-1][DANGER]/ 255.0 if agent.position[1] > 0 else 0
            observations[agent.id]['grid_perception'][1][WALK] = self.grid[agent.position[0], agent.position[1]-1][WALK]/ 255.0 if agent.position[1] > 0 else 0
            observations[agent.id]['grid_perception'][1][FOOD] = self.grid[agent.position[0], agent.position[1]-1][FOOD]/ 255.0 if agent.position[1] > 0 else 0
            observations[agent.id]['grid_perception'][2][DANGER] = self.grid[agent.position[0]+1, agent.position[1]][DANGER]/ 255.0 if agent.position[0] < self.grid_size[0]-1 else 0
            observations[agent.id]['grid_perception'][2][WALK] = self.grid[agent.position[0]+1, agent.position[1]][WALK]/ 255.0 if agent.position[0] < self.grid_size[0]-1 else 0
            observations[agent.id]['grid_perception'][2][FOOD] = self.grid[agent.position[0]+1, agent.position[1]][FOOD]/ 255.0 if agent.position[0] < self.grid_size[0]-1 else 0
            observations[agent.id]['grid_perception'][3][DANGER] = self.grid[agent.position[0], agent.position[1]+1][DANGER]/ 255.0 if agent.position[1] < self.grid_size[1]-1 else 0
            observations[agent.id]['grid_perception'][3][WALK] = self.grid[agent.position[0], agent.position[1]+1][WALK]/ 255.0 if agent.position[1] < self.grid_size[1]-1 else 0
            observations[agent.id]['grid_perception'][3][FOOD] = self.grid[agent.position[0], agent.position[1]+1][FOOD]/ 255.0 if agent.position[1] < self.grid_size[1]-1 else 0
            observations[agent.id]['grid_perception'][4][DANGER] = self.grid[agent.position[0]-1, agent.position[1]][DANGER]/ 255.0 if agent.position[0] > 0 else 0
            observations[agent.id]['grid_perception'][4][WALK] = self.grid[agent.position[0]-1, agent.position[1]][WALK]/ 255.0 if agent.position[0] > 0 else 0
            observations[agent.id]['grid_perception'][4][FOOD] = self.grid[agent.position[0]-1, agent.position[1]][FOOD]/ 255.0 if agent.position[0] > 0 else 0
            observations[agent.id]['average_signal_strength'] = avg_signal_stregth # TODO normalize
            observations[agent.id]['average_signal_type'] = avg_signal_type
            observations[agent.id]['average_signal_direction_x'] = avg_signal_direction_x
            observations[agent.id]['average_signal_direction_y'] = avg_signal_direction_y
            observations[agent.id]['closest_signal_strength'] = strongest_signal_strength
            observations[agent.id]['closest_signal_type'] = strongest_signal_type
            observations[agent.id]['closest_signal_direction_x'] = strongest_signal_direction_x
            observations[agent.id]['closest_signal_direction_y'] = strongest_signal_direction_y
            observations[agent.id]['energy'] = agent.attributes.energy / agent.attributes.max_energy

        # check for terminations
        terminations = {a.id: a.attributes.energy <= 0 for a in self.agents}
        # picks random agent and sets its termination to True
        # terminations[np.random.choice(self.agents)] = True

        rewards = {a.id: 0.0 for a in self.agents}
        for agent in self.agents:
            rewards[agent.id] += received_signals * 0.1
            # if agent.attributes.energy > agent.attributes.max_energy/2:
            #     rewards[agent.id] += 1
            if agent.attributes.energy > agent.attributes.max_energy/4:
                rewards[agent.id] += 0.1
            if not terminations[agent.id]:
                rewards[agent.id] += 10
        # check for truncations
        truncations = {a.id: False for a in self.agents}
        self.iteration += 1
        if self.iteration >= self.max_iterations:
            truncations = {a.id: True for a in self.agents}
            for agent in self.agents:
                rewards[agent.id] += 100

        
        
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
            color_value = int(agent.attributes.energy/agent.attributes.max_energy*255)
            if color_value > 255:
                color_value = 255
            elif color_value < 0:
                color_value = 0
            pygame.draw.circle(
                canvas,
                (color_value, color_value, color_value),
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
    GRID
    FOOD CONTENT 
    AVERAGE_SIGNAL_STRENGTH
    AVERAGE_SIGNAL_TYPE
    AVERAGE_SIGNAL_DIRECTION_X
    AVERAGE_SIGNAL_DIRECTION_Y
    CLOSEST_SIGNAL_STRENGTH 
    CLOSEST_SIGNAL_TYPE
    CLOSEST_SIGNAL_DIRECTION_X
    CLOSEST_SIGNAL_DIRECTION_Y
    ENERGY
    '''

    # @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_id):
        if hasattr(self, 'observation_spaces') is False:
            self.observation_spaces = dict()
        if agent_id not in self.observation_spaces or self.observation_spaces[agent_id] is None:
            self.observation_spaces[agent_id] = Dict({
                'grid_perception': Box(low=0, high=1, shape=(5,3), dtype=np.float32),
                'average_signal_strength': Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'average_signal_type': Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'average_signal_direction_x': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'average_signal_direction_y': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'closest_signal_strength': Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'closest_signal_type': Box(low=0, high=1, shape=(1,), dtype=np.float32),
                'closest_signal_direction_x': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'closest_signal_direction_y': Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'energy': Box(low=0, high=1, shape=(1,), dtype=np.float32),
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
        DROP_FOOD
    BOX ACTIONS:
        SIGNAL: STRENGTH TYPE
    '''

    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent_id):
        if hasattr(self, 'action_spaces') is False:
            self.action_spaces = dict()
        if agent_id not in self.action_spaces or self.action_spaces[agent_id] is None:
            self.action_spaces[agent_id] = Tuple((Discrete(8,), Box(low=0, high=1, shape=(2,), dtype=np.float32)))
        return self.action_spaces[agent_id]
