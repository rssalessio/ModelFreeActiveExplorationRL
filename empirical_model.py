import numpy as np
from itertools import product
from typing import List, Tuple, NamedTuple, Optional, Set
from enum import Enum
from maze import Coordinate
from numpy.typing import NDArray

class EmpiricalModel(object):

    def __init__(self, num_states: int, num_actions: int):
        """
        Transition function model for a grids

        Args:
            num_states (int): Number of states
            num_actions (int): Number of actions
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_visits_actions = np.zeros(shape=(num_states, num_actions, num_states), dtype=np.int64)
        self.transition_function = np.ones_like(self.num_visits_actions, dtype=np.float64) / num_states
        self.reward = np.zeros_like(self.num_visits_actions, dtype=np.float64)

    def update_visits(self, from_state: int, action: int, to_state: int, reward: float):
        """Updates the transition function given an experience

        Args:
            from_state (Coordinate): state s
            action (int): action a
            to_state (Coordinate): next state s'
            reward (float): ereward
        """
        self.num_visits_actions[from_state, action, to_state] += 1
        self.reward[from_state, action, to_state] = reward

        self.transition_function[from_state, action] = (
            self.num_visits_actions[from_state, action] / self.num_visits_actions[from_state, action].sum()
        )

    