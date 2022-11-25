import numpy as np
from itertools import product
from typing import List, Tuple, NamedTuple, Optional, Set
from enum import Enum
from maze import Coordinate

class Model(object):

    def __init__(self, num_rows: int, num_columns: int, num_actions: int):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.num_actions = num_actions
        self.num_visits_actions = np.zeros((num_rows, num_columns, num_actions))
        self.transition_function = np.zeros((num_rows, num_columns, num_actions, num_rows, num_columns))

    def update_visits(self, from_state: Coordinate, action: int, to_state: Coordinate):
        self.num_visits_actions[from_state.x, from_state.y, action, to_state.x, to_state.y] += 1
