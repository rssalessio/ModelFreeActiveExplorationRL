import numpy as np
from itertools import product
from typing import List, Tuple, NamedTuple, Optional, Set
from enum import Enum
from maze import Coordinate
from numpy.typing import NDArray

class EmpiricalModel(object):

    def __init__(self, num_rows: int, num_columns: int, num_actions: int):
        """
        Transition function model for a grids

        Args:
            num_rows (int): Number of rows in the maze
            num_columns (int): Number of columns in the maze
            num_actions (int): Number of actions
        """
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_actions = num_actions
        self.num_visits_actions = np.zeros(shape=(num_rows * num_columns, num_actions, num_rows * num_columns), dtype=np.int64)
        self.transition_function = np.ones_like(self.num_visits_actions) / (num_rows * num_columns)

    def to_id(self, pt: Coordinate) -> int:
        """Transforms a coordinate into an intger

        Args:
            pt (Coordinate): coordinate

        Returns:
            int: corresponding id
        """        
        return pt.x + self.num_columns * pt.y

    def update_visits(self, from_state: Coordinate, action: int, to_state: Coordinate):
        """Updates the transition function given an experience

        Args:
            from_state (Coordinate): state s
            action (int): action a
            to_state (Coordinate): next state s'
        """        
        from_state = self.to_id(from_state)
        to_state = self.to_id(to_state)
        
        self.num_visits_actions[from_state, action, to_state] += 1

        _temp: NDArray[np.int64] = self.num_visits_actions[from_state, action]
        self.transition_function[from_state, action] = _temp / _temp.sum(-1)

    