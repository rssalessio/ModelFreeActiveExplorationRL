import numpy as np
from itertools import product
from typing import List, Tuple, NamedTuple, Optional, Set
from enum import Enum

class Coordinate(NamedTuple):
    x: int
    y: int

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Maze(object):
    initial_position: Coordinate = Coordinate(0, 0)
    current_position: Coordinate = Coordinate(0, 0)

    def __init__(self, n_rows: int, n_columns: int, failure_probability: Optional[float] = 0, walls: Optional[Set[Coordinate]] = None, random_walls: bool = False, fraction_walls: float = 0.1):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.done_position = Coordinate(self.n_columns - 1, self.n_rows - 1)
        self.failure_probability = failure_probability

        assert failure_probability <= 1 and failure_probability >= 0, 'Failure probability should be in [0,1]'

        if not random_walls:
            self.walls = walls if walls else set()
            # @todo check that the walls are within the boundaries
        else:
            assert fraction_walls > 0 and fraction_walls < 1, 'Fraction of walls needs to be a real number in (0,1)'
            possible_positions = list(product(range(n_columns), range(n_rows)))
            positions = np.random.choice(len(possible_positions), size = int(len(possible_positions) * fraction_walls), replace=False)
            self.walls = set([Coordinate(*possible_positions[x]) for x in positions])
        
        if self.initial_position in self.walls:
            self.walls.remove(self.initial_position)
        if self.done_position in self.walls:
            self.walls.remove(self.done_position)

    def reset(self) -> Coordinate:
        self.current_position = self.initial_position
        return self.current_position    

    def step(self, action: Action) -> Tuple[Coordinate, float, bool]:
        pos = self.current_position
        done = False
        reward = 0
        
        if self.current_position == self.done_position:
            # Absorbing state
            done = True
        elif np.random.uniform() < 1-self.failure_probability:
            match action:
                case Action.UP:
                    if pos.y != self.n_rows - 1 and Coordinate(pos.x, pos.y + 1) not in self.walls:
                        self.current_position = Coordinate(pos.x, pos.y + 1)
                case Action.RIGHT:
                    if pos.x != self.n_columns - 1 and Coordinate(pos.x + 1, pos.y) not in self.walls:
                        self.current_position = Coordinate(pos.x + 1, pos.y)
                case Action.DOWN:
                    if pos.y != 0 and Coordinate(pos.x, pos.y - 1) not in self.walls:
                        self.current_position = Coordinate(pos.x, pos.y - 1)
                case Action.LEFT:
                    if pos.x != 0 and Coordinate(pos.x - 1, pos.y) not in self.walls:
                        self.current_position = Coordinate(pos.x - 1, pos.y)
                case _:
                    raise ValueError(f'Invalid action {action}')

            if self.current_position == self.done_position:
                reward = 1
                done = True
        
        return self.current_position, reward, done

maze = Maze(3,3, failure_probability=1,random_walls=True, fraction_walls=0.5)
print(maze.walls)
print(maze.step(Action.DOWN))
print(maze.step(Action.LEFT))
print(maze.step(Action.RIGHT))
print(maze.step(Action.UP))