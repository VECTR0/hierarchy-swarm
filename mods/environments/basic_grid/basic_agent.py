from collections import namedtuple
from math import e
from numpy import number


class BasicAgent:
    def __init__(self, position : 'tuple[int, int]', id : str, energy : float = 100):
        self.position = position
        self.id = id
        self.attributes = namedtuple("attributes", [
            "energy", 
            "max_energy",
            "movement_energy_cost",
            "signal_emit_range",
            "signal_emit_cost",
            "signal_hear_range",
            "idle_energy_cost",
            ])
        self.attributes.energy = energy # type: ignore
        self.attributes.max_energy = energy # type: ignore
        self.attributes.movement_energy_cost = 1 # type: ignore
        self.attributes.signal_emit_range = 10 # type: ignore
        self.attributes.signal_hear_range = 50 # type: ignore
        self.attributes.signal_emit_cost = 1 # type: ignore
        self.attributes.idle_energy_cost = 0.1 # type: ignore