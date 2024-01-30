from abc import ABC, abstractmethod
import numpy as np
'''
OrderedDict([
('average_signal_direction_x', array([-0.7229175], dtype=float32)),
('average_signal_direction_y', array([-0.49689758], dtype=float32)),
('average_signal_strength', array([0.26397312], dtype=float32)),
('average_signal_type', array([0.66225153], dtype=float32)),
('closest_signal_direction_x', array([0.9393185], dtype=float32)),
('closest_signal_direction_y', array([-0.05889778], dtype=float32)),
('closest_signal_strength', array([0.11937959], dtype=float32)),
('closest_signal_type', array([0.1261529], dtype=float32)),
('energy', array([0.9405555], dtype=float32)),
('grid_perception', array([[0.41049933, 0.46692604, 0.00674017],
[0.4818051 , 0.22084613, 0.43405038],
[0.425594  , 0.01025833, 0.2962167 ],
[0.50496334, 0.42665723, 0.2829967 ],
[0.09774695, 0.49640056, 0.9392032 ]], dtype=float32))
])
'''
class MLPBrain(ABC):
    def __init__(self, layers, seed=None, id = None):
        self.id = id
        self.init_structure(layers=layers, seed=seed)
    
    def act(self, observation, environment):
        for i in range(len(observation["grid_perception"][0])):
                self.input[i * 3 + 0] = observation["grid_perception"][i][0]
                self.input[i * 3 + 1] = observation["grid_perception"][i][1]
                self.input[i * 3 + 2] = observation["grid_perception"][i][2]
        self.input[15] = observation["energy"]
        self.input[16] = observation["average_signal_strength"]
        self.input[17] = observation["average_signal_type"]
        self.input[18] = observation["closest_signal_strength"]
        self.input[19] = observation["closest_signal_type"]
        self.input[20] = observation["average_signal_direction_x"]
        self.input[21] = observation["average_signal_direction_y"]
        self.input[22] = observation["closest_signal_direction_x"]
        self.input[23] = observation["closest_signal_direction_y"]

        
        input = self.input
        input = [[x] for x in input]
        for i in range(1,len(self.layers)):
            input = np.matmul(self.structure["weigths"][i-1], np.array(input)) + self.structure["biases"][i-1]
            input = 1/(1+np.exp(-input))
        input = [x[0] for x in input]
        #pick max from first 4 elements of input
        max_index = 0
        max_value = input[0]
        for i in range(8):
            if input[i] > max_value:
                max_value = input[i]
                max_index = i
        action = [max_index, [input[8],input[9]]]
        return action
    
    def clone(self):
        return MLPBrain.deserialize(self.serialize())

    def reset(self):
        pass

    def serialize(self):
        ret = {
            "id": self.id,
            "layers": self.layers,
            "weights": [x.tolist() for x in self.structure["weigths"]],
            "biases": [x.tolist() for x in self.structure["biases"]]
        }
        return ret

    @staticmethod
    def deserialize(data):
        ret = MLPBrain(id=data["id"], layers=data["layers"])
        ret.structure["weigths"] = [np.array(x) for x in data["weights"]]
        ret.structure["biases"] = [np.array(x) for x in data["biases"]]
        return ret

    def mutate(self, other):
        random_layer_index = np.random.randint(len(self.layers)-1)
        random_layer = self.structure["weigths"][random_layer_index]
        xy = np.random.randint(random_layer.shape[0]), np.random.randint(random_layer.shape[1])
        random_layer[xy] += (np.random.random() * 2 - 1) * 0.1

        random_bias = self.structure["biases"][random_layer_index]
        random_bias[0] += (np.random.random() * 2 - 1) * 0.1
        
    def init_structure(self, layers, seed=None):
        np.random.seed(seed)
        self.layers = layers 
        self.input = [0] * layers[0]
        self.structure = {
            "weigths": [],
            "biases": []
        }
        for i in range(1,len(layers)):
            input_size, output_size = layers[i-1], layers[i]
            self.structure["weigths"].append(np.random.normal(loc=0, scale=0.1, size=(output_size, input_size)))
            self.structure["biases"].append(np.zeros((output_size, 1)))

