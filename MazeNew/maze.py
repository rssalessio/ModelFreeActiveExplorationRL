from __future__ import annotations
import numpy as np
from itertools import product
from typing import List, Tuple, NamedTuple, Optional, Set, Dict, Union
from numpy.typing import NDArray
from enum import Enum

class Coordinate(NamedTuple):
    row: int
    col: int
    num_rows: int
    num_cols: int

    def __str__(self) -> str:
        return f'({self.row},{self.col})'
    
    def copy(self):
        return Coordinate(self.row, self.col, self.num_rows, self.num_cols)
    
    def is_close(self, p: Coordinate) -> bool:
        if self.col == p.col: return abs(self.row - p.row) <= 1
        elif self.row == p.row: return abs(self.col - p.col) <= 1
        return False
    
    def distance(self, p: Coordinate) -> int:
        return np.abs(self.col - p.col) + np.abs(self.row - p.row)
    
    @property
    def UP(self):
        return Coordinate(max(0, self.row - 1), self.col, self.num_rows, self.num_cols)
    
    @property
    def RIGHT(self):
        return Coordinate(self.row, min(self.num_cols - 1, self.col + 1), self.num_rows, self.num_cols)
    
    @property
    def DOWN(self):
        return Coordinate(min(self.num_rows - 1, self.row + 1), self.col, self.num_rows, self.num_cols)
    
    @property
    def LEFT(self):
        return Coordinate(self.row, max(0, self.col - 1), self.num_rows, self.num_cols)


class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __int__(self):
        return self.value

class Maze(object):
    def __init__(self,
                 block_width: int = 4,
                 block_height: int = 3,
                 slippery_probability: float = 0.3,
                 movement_reward: float = 0.01,
                 exit_reward_easy: float = 0.5,
                 exit_reward_hard: float = 1,
                 observe_entire_grid: bool = False,
                 time_limit: bool = False):
        self._block_width = block_width
        self._block_height = block_height
        self._num_states = (3 + block_width * 2) + (3 + block_height * 2)
        self._n_rows = 3 + block_height * 2
        self._n_columns = 3 + block_width * 2
        self._slippery_probability = slippery_probability
        
        self._time_limit = time_limit
        self._time_limit_amount = self._n_rows * self._n_columns
        
        self._movement_reward = movement_reward
        self._exit_reward_easy = exit_reward_easy
        self._exit_reward_hard = exit_reward_hard
        
        self._done_position = Coordinate(block_height + 1, block_width + 1, num_rows=self._n_rows, num_cols=self._n_columns)
        self._observation_space = set(product(range(self._n_rows), range(self._n_columns)))
        self._states_mapping: Dict[Coordinate, int] = {}
        self._id_to_coord_mapping: Dict[int, Coordinate] = {}
        
        
        self._walls = []
        # Remove walls and define mapping betweeen cordinates and integers
        for _, coord in enumerate(self._observation_space):
            if self.is_wall(Coordinate(*coord, num_rows=self._n_rows, num_cols=self._n_columns)):
                self._walls.append(coord)

        for wall in self._walls:
            self._observation_space.remove(wall)
        
        for id, coord in enumerate(self._observation_space):
            coord = Coordinate(*coord, num_rows=self._n_rows, num_cols=self._n_columns)
            self._states_mapping[coord] = id
            self._id_to_coord_mapping[id] = coord

        self._observe_entire_grid = observe_entire_grid
        self._current_position: Coordinate = Coordinate(0, 0, num_rows=self._n_rows, num_cols=self._n_columns)
        
        self._t = 0
        
        self._build_transition_probabilities()
        self._build_rewards()
        
    @property
    def reward_function(self) -> NDArray[np.float64]:
        return self._rewards
    
    @property
    def transition_function(self) -> NDArray[np.float64]:
        return self._transition_function
    
    def _build_transition_probabilities(self):
        """Build the transition probabilit yfunction, of the form (state, action, next_state)
        """        
        S_dim = len(self.observation_space)
        self._transition_function = np.zeros((S_dim, 4, S_dim))
        for s, coord_state in enumerate(self._observation_space):
            coord_state = Coordinate(*coord_state, num_rows=self._n_rows, num_cols=self._n_columns)
            if self.is_wall(coord_state): continue
            if coord_state == self._done_position:
                for action in Action: self._transition_function[s, action.value, s] = 1
                continue
                
            for s_next, coord_state_next in enumerate(self._observation_space):
                coord_state_next = Coordinate(*coord_state_next, num_rows=self._n_rows, num_cols=self._n_columns)
                if self.is_wall(coord_state_next): continue
                if coord_state.is_close(coord_state_next) is False: continue
                
                for possible_next_state, possible_action in [
                    (coord_state.UP, Action.UP), (coord_state.DOWN, Action.DOWN),
                    (coord_state.LEFT, Action.LEFT), (coord_state.RIGHT, Action.RIGHT)]:
                    
                    # if coord_state.row == 8 and coord_state.col == 0:
                    #     import pdb
                    #     pdb.set_trace()
                    _s_next = s_next
                    if self.is_wall(possible_next_state) or possible_next_state == coord_state:
                        possible_next_state = coord_state
                        _s_next = s
                    
                    if possible_next_state == coord_state_next:
                        if self._done_position.distance(coord_state_next) < self._done_position.distance(coord_state):
                            self._transition_function[s, possible_action.value, _s_next] = 1 - self._slippery_probability
                            self._transition_function[s, possible_action.value, s] = self._slippery_probability
                        else:
                            self._transition_function[s, possible_action.value, _s_next] = 1
        
        self._transition_function = self._transition_function / self._transition_function.sum(-1, keepdims=True)
    
    def _build_rewards(self):
        """
        Build the reward function of the form (state, action, next_state)
        """        
        S_dim = len(self.observation_space)
        self._rewards = np.zeros((S_dim, 4))
        
        # NB We are assuming that these positions are not occupied by walls!
        pos_left = self._done_position.copy().LEFT
        pos_up = self._done_position.copy().UP
        pos_down = self._done_position.copy().DOWN
        pos_right = self._done_position.copy().RIGHT
        
        for pos, action in [(pos_left, Action.RIGHT.value), (pos_up, Action.DOWN.value)]:
            self._rewards[
                self._states_mapping[pos],
                action
            ] = (1 - self._slippery_probability) * self._exit_reward_easy
        
        for pos, action in [(pos_right, Action.LEFT.value), (pos_down, Action.UP.value)]:
            self._rewards[
                self._states_mapping[pos],
                action
            ] = (1 - self._slippery_probability) * self._exit_reward_hard
        
        for s, coord in enumerate(self._observation_space):
            coord_state = Coordinate(*coord, num_rows=self._n_rows, num_cols=self._n_columns)
            if self.is_wall(coord_state): continue
            for action, coord_next_state in [
                (Action.UP.value, coord_state.UP), (Action.DOWN.value, coord_state.DOWN),
                (Action.LEFT.value, coord_state.LEFT),(Action.RIGHT.value, coord_state.RIGHT)]:
                if self.is_wall(coord_next_state):
                    coord_next_state = coord_state
                if self._done_position.distance(coord_next_state) >= self._done_position.distance(coord_state):
                    self._rewards[s, action] = self._movement_reward / (np.sqrt(self._n_rows  * self._n_columns))
       

    def is_wall(self, p: Coordinate) -> bool:
        if  p.col in [0, self._block_width + 1, 2 * self._block_width + 2]:
            return False
        if  p.row in [0, self._block_height + 1, 2 * self._block_height + 2]:
            return False
        return True
        
        
    def map_coordinate(self, p: Coordinate, observe_grid: bool = False) -> Union[int, NDArray[np.float64]]:
        if observe_grid:
            grid = np.zeros((self._n_rows, self._n_columns))
            grid[p.row, p.col] = 1
            return grid
        return self._states_mapping[self._current_position] 
    
    @property
    def observation_space(self):
        return self._observation_space

    def reset(self) -> Union[int, NDArray[np.float64]]:
        self._current_position: Coordinate = Coordinate(0, 0, num_rows=self._n_rows, num_cols=self._n_columns)
        self._t = 0
        return self.map_coordinate(self._current_position, self._observe_entire_grid)

    def step(self, action: Action) -> Tuple[Union[int, NDArray[np.float64]], float, bool]:
        pos = self._current_position
        done = False
        reward = 0
        self._t += 1
        
        if self._current_position == self._done_position or (self._time_limit and self._t >= self._time_limit_amount):
            # Absorbing state
            done = True
        else:
            new_position = self._current_position.copy()
            match action:
                case Action.UP:
                    if pos.row > 0 and pos.col in [0, self._block_width + 1, 2 * self._block_width + 2]:
                        new_position = pos.UP
                case Action.RIGHT:
                    if pos.col < self._n_columns - 1 and pos.row in [0, self._block_height + 1, 2 * self._block_height + 2]:
                        new_position = pos.RIGHT
                case Action.DOWN:
                    if pos.row < self._n_rows - 1 and pos.col in [0, self._block_width + 1, 2 * self._block_width + 2]:
                        new_position = pos.DOWN
                case Action.LEFT:
                    if pos.col > 0 and pos.row in [0, self._block_height + 1, 2 * self._block_height + 2]:
                        new_position = pos.LEFT
                case _:
                    raise ValueError(f'Invalid action {action}')
            
            self.prev_position = self._current_position.copy()
            reward = 0

            if self._done_position.distance(new_position) < self._done_position.distance(self._current_position):
                self._current_position = self._current_position if np.random.uniform() < self._slippery_probability else new_position
            elif self._current_position != self._done_position:
                reward = self._movement_reward / (np.sqrt(self._n_rows  * self._n_columns))
                self._current_position = new_position
                
            
            if self._current_position == self._done_position:
                reward = 0
                done = True
                
                if self.prev_position != self._done_position:
                    theta = 0
                    if  self.prev_position.row < self._done_position.row:
                        theta = self._exit_reward_easy
                    elif self.prev_position.row > self._done_position.row:
                        theta = self._exit_reward_hard
                    elif self.prev_position.col < self._done_position.col:
                        theta = self._exit_reward_easy
                    elif self.prev_position.col > self._done_position.col:
                        theta = self._exit_reward_hard
                    reward = 1 if np.random.uniform() < theta else 0
        
        return self.map_coordinate(self._current_position, self._observe_entire_grid), reward, done

    def show(self, policy: Optional[NDArray[np.float64]] = None, value: Optional[NDArray[np.float64]] = None):
        c = 65

        def _format_cell(pos):
            if self.is_wall(pos):
                return 'X'
            
            if policy is None and value is None:
                if pos == self._current_position:
                    return 'P'
                elif pos == self._done_position:
                    return 'E'
                return ' '
            else:
                if value is not None:
                    return str(np.round(value[self._states_mapping[pos]], 2))
                else:
                    match Action(policy[self._states_mapping[pos]]):
                        case Action.UP: return '↑'
                        case Action.DOWN: return '↓'
                        case Action.LEFT: return '←'
                        case Action.RIGHT: return '→'

        # First row
        print(f"  ", end='')
        for j in range(self._n_columns):
            print(f"| {j} ", end='')
        print("| ")
        print((self._n_columns*4+4)*"-")

        # Other rows
        for i in range(self._n_rows):
            print(f"{chr(c+i)} ", end='')
            for j in range(self._n_columns):
                print(f"| {_format_cell(Coordinate(i, j, self._n_rows, self._n_columns))} ", end='')
            print("| ")
            print((self._n_columns*4+4)*"-")



if __name__ == '__main__':
    env = Maze(observe_entire_grid=False)
    state = env.reset()
    state, reward, done = env.step(Action.RIGHT)
    state, reward, done = env.step(Action.RIGHT)
    state, reward, done = env.step(Action.RIGHT)
    state, reward, done = env.step(Action.DOWN)
    state, reward, done = env.step(Action.RIGHT)
    state, reward, done = env.step(Action.DOWN)
    state, reward, done = env.step(Action.RIGHT)
    state, reward, done = env.step(Action.DOWN)
    state, reward, done = env.step(Action.RIGHT)
    state, reward, done = env.step(Action.DOWN)
    state, reward, done = env.step(Action.RIGHT)
    state, reward, done = env.step(Action.DOWN)
    state, reward, done = env.step(Action.RIGHT)
    state, reward, done = env.step(Action.DOWN)
    print(reward)
    state, reward, done = env.step(Action.RIGHT)
    print(reward)
    state, reward, done = env.step(Action.DOWN)
    print(state)
    print(reward)
    from utils import policy_iteration
    V,pi,Q = policy_iteration(0.99, env.transition_function, env.reward_function[..., np.newaxis])
    actions =  [f'{env._id_to_coord_mapping[s]}: {pi[s]}' for s in env._id_to_coord_mapping.keys()] 
    print(actions)
    env.show(pi)
    