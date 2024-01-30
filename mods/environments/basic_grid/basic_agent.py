from attr import dataclass
import random


class BasicAgent:
    def __init__(self, position : 'tuple[int, int]', id : str, energy : float = 1):
        self.position = position
        self.id = id
        # self.attributes = AgentAttributes(energy, energy, 0.25, 50, 1, 20, 0.1)
        # self.attributes = AgentAttributes(energy=energy, max_energy=energy, movement_energy_cost=0.05, signal_emit_range=1, signal_emit_cost=1, signal_hear_range=1, idle_energy_cost=0.1)
        self.attributes = BasicAgent.generate_random_attributes()
        
    @staticmethod
    def generate_random_attributes():
        max_energy = random.uniform(0, 1)
        energy = max_energy
        signal_emit_range = random.uniform(0, 1)
        signal_emit_cost = 0
        signal_hear_range = random.uniform(0, 1)
        movement_energy_cost = 0
        idle_energy_cost = 0
        ret = AgentAttributes(energy=energy, max_energy=max_energy, movement_energy_cost=movement_energy_cost, signal_emit_range=signal_emit_range, signal_emit_cost=signal_emit_cost, signal_hear_range=signal_hear_range, idle_energy_cost=idle_energy_cost)
        BasicAgent.calculate_costs(ret)
        return ret

    @staticmethod
    def mutate(agent_attributes):  # TODO
        random_attribute = random.choice(list(vars(agent_attributes).keys()))
        current_attribute_value = getattr(agent_attributes, random_attribute)
        new_attribute_value = current_attribute_value + random.uniform(-0.1, 0.1)
        setattr(agent_attributes, random_attribute, new_attribute_value)
        agent_attributes.energy = agent_attributes.max_energy
        BasicAgent.calculate_costs(agent_attributes)

    @staticmethod
    def calculate_costs(agent_attributes):
        BasicAgent.clip_attributes(agent_attributes)
        aa = agent_attributes
        aa.movement_energy_cost = aa.max_energy * 0.01 
        aa.signal_emit_cost = aa.signal_emit_range * 0.007
        aa.idle_energy_cost = aa.signal_hear_range * 0.1 + 0.01

    @staticmethod
    def clip_attributes(agent_attributes):
        aa = agent_attributes
        aa.max_energy = max(0, min(aa.max_energy, 1))
        aa.signal_emit_range = max(0, min(aa.signal_emit_range, 1))
        aa.signal_hear_range = max(0, min(aa.signal_hear_range, 1))

@dataclass
class AgentAttributes:
    energy : float
    max_energy : float
    movement_energy_cost : float
    signal_emit_range : float
    signal_emit_cost : float
    signal_hear_range : float
    idle_energy_cost : float

    def serialize(self):
        return {
            'energy': self.energy,
            'max_energy': self.max_energy,
            'movement_energy_cost': self.movement_energy_cost,
            'signal_emit_range': self.signal_emit_range,
            'signal_emit_cost': self.signal_emit_cost,
            'signal_hear_range': self.signal_hear_range,
            'idle_energy_cost': self.idle_energy_cost
        }

    @staticmethod
    def deserialize(data):
        return AgentAttributes(
            energy=data['energy'],
            max_energy=data['max_energy'],
            movement_energy_cost=data['movement_energy_cost'],
            signal_emit_range=data['signal_emit_range'],
            signal_emit_cost=data['signal_emit_cost'],
            signal_hear_range=data['signal_hear_range'],
            idle_energy_cost=data['idle_energy_cost']
        )