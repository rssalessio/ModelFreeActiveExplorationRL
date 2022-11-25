import numpy as np
from itertools import product
from typing import List, Tuple, NamedTuple, Optional, Set
from enum import Enum
from maze import Coordinate



class Model(object):

    def __init__(self, num_rows: int, num_columns: int, num_actions: int):
        """
        Transition function model for a maze

        Args:
            num_rows (int): Number of rows in the maze
            num_columns (int): Number of columns in the maze
            num_actions (int): Number of actions
        """
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_actions = num_actions
        self.num_visits_actions = np.zeros((num_rows * num_columns, num_actions))
        self.transition_function = np.zeros((num_rows * num_columns, num_actions, num_rows * num_columns))

    def to_id(self, pt: Coordinate) -> int:
        return pt.x + self.num_columns * pt.y

    def update_visits(self, from_state: Coordinate, action: int, to_state: Coordinate):
        from_state = self.to_id(from_state)
        to_state = self.to_id(to_state)
        self.num_visits_actions[from_state, action, to_state] += 1

        _temp = self.num_visits_actions[from_state, action]
        self.num_visits_actions[from_state, action] = _temp / _temp.sum(-1)[:, None]

    