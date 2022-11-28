import numpy as np
from itertools import product
from typing import List, Tuple, NamedTuple, Optional, Set
from enum import Enum

class Coordinate(NamedTuple):
    x: int
    y: int

    def __str__(self) -> str:
        return f'({self.x},{self.y})'

class MazeParameters(NamedTuple):
    num_rows: int
    num_columns: int
    slippery_probability: Optional[float] = 0,
    walls: Optional[List[Coordinate]] = None,
    random_walls: bool = False,
    fraction_walls: float = 0.1

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

class Maze(object):
    initial_position: Coordinate = Coordinate(0, 0)
    current_position: Coordinate = Coordinate(0, 0)

    def __init__(self, parameters: MazeParameters):
        self.n_rows = parameters.num_rows
        self.n_columns = parameters.num_columns
        self.done_position = Coordinate(self.n_columns - 1, self.n_rows - 1)
        self.slippery_probability = parameters.slippery_probability
        self._observation_space = set(product(range(self.n_columns), range(self.n_rows)))

        assert parameters.slippery_probability <= 1 and parameters.slippery_probability >= 0, 'Failure probability should be in [0,1]'

        if not parameters.random_walls:
            self.walls = set(parameters.walls) if parameters.walls else set()
            # @todo check that the walls are within the boundaries
        else:
            assert parameters.fraction_walls > 0 and parameters.fraction_walls < 1, 'Fraction of walls needs to be a real number in (0,1)'
            positions = np.random.choice(len(self.observation_space), size = int(len(self.observation_space) * parameters.fraction_walls), replace=False)
            self.walls = set([Coordinate(*self.observation_space[x]) for x in positions])
        
        if self.initial_position in self.walls:
            self.walls.remove(self.initial_position)
        if self.done_position in self.walls:
            self.walls.remove(self.done_position)
            
        for x in self.walls:
            if x in self.observation_space:
                self.observation_space.remove(x)

    @property
    def observation_space(self):
        return self._observation_space

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
        else:
            if np.random.uniform() < self.slippery_probability:
                if np.random.uniform() < 0.5:
                    return self.current_position, reward, done
                elif action == Action.UP or action == Action.DOWN:
                    action = Action.LEFT if np.random.uniform() < 0.5 else Action.RIGHT
                else:
                    action = Action.UP if np.random.uniform() < 0.5 else Action.DOWN

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
    
    def show(self):
        c = 65

        def _format_cell(pos, current_position, walls, done_position):
            if pos in walls:
                return 'X'
            elif pos == current_position:
                return 'P'
            elif pos == done_position:
                return 'E'
            return ' '

        # First row
        print(f"  ", end='')
        for j in range(self.n_columns):
            print(f"| {j} ", end='')
        print("| ")
        print((self.n_columns*4+4)*"-")

        # Other rows
        for i in range(self.n_rows-1,-1,-1):
            print(f"{chr(c+i)} ", end='')
            for j in range(self.n_columns):
                print(f"| {_format_cell((j,i), self.current_position, self.walls, self.done_position)} ", end='')
            print("| ")
            print((self.n_columns*4+4)*"-")


if __name__ == '__main__':
    maze = Maze(MazeParameters(20,20, slippery_probability=.1,random_walls=True, fraction_walls=0.1))
    print(maze.walls)
    print(maze.step(Action.DOWN))
    print(maze.step(Action.LEFT))
    print(maze.step(Action.RIGHT))
    print(maze.step(Action.UP))

    maze.show()