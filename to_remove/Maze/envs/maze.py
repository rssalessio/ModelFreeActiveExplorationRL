import numpy as np
from itertools import product
from typing import List, Tuple, NamedTuple, Optional, Set, Dict
from numpy.typing import NDArray
from enum import Enum

class Coordinate(NamedTuple):
    x: int
    y: int

    def __str__(self) -> str:
        return f'({self.x},{self.y})'
    
    def copy(self):
        return Coordinate(self.x, self.y)
    
    def is_close(self, p) -> bool:
        if self.x == p.x: return abs(self.y - p.y) <= 1
        elif self.y == p.y: return abs(self.x - p.x) <= 1
        return False
    
    @property
    def UP(self):
        return Coordinate(self.x, self.y + 1)
    
    @property
    def RIGHT(self):
        return Coordinate(self.x + 1, self.y)
    
    @property
    def DOWN(self):
        return Coordinate(self.x, self.y - 1)
    
    @property
    def LEFT(self):
        return Coordinate(self.x - 1, self.y)

class MazeParameters(NamedTuple):
    num_rows: int
    num_columns: int
    slippery_probability: Optional[float] = 0,
    walls: Optional[List[Coordinate]] = None,
    random_walls: bool = False,
    fraction_walls: float = 0.1
    stay_probability_after_slipping: Optional[float] = 0.5

class Action(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __int__(self):
        return self.value

class Maze(object):
    initial_position: Coordinate = Coordinate(0, 0)
    current_position: Coordinate = Coordinate(0, 0)

    def __init__(self, parameters: MazeParameters):
        self.n_rows = parameters.num_rows
        self.n_columns = parameters.num_columns
        self.done_position = Coordinate(self.n_columns - 1, self.n_rows - 1)
        self.slippery_probability = parameters.slippery_probability
        self.stay_probability_after_slipping = parameters.stay_probability_after_slipping
        self._observation_space = set(product(range(self.n_columns), range(self.n_rows)))
        self._states_mapping: Dict[Coordinate, int] = {}

        assert parameters.slippery_probability <= 1 and parameters.slippery_probability >= 0, 'Failure probability should be in [0,1]'

        if not parameters.random_walls:
            self.walls = set(parameters.walls) if parameters.walls else set()
            # @todo check that the walls are within the boundaries
        else:
            assert parameters.fraction_walls > 0 and parameters.fraction_walls < 1, 'Fraction of walls needs to be a real number in (0,1)'
            positions = np.random.choice(len(self._observation_space), size = int(len(self._observation_space) * parameters.fraction_walls), replace=False)
            self.walls = set([Coordinate(*self._observation_space[x]) for x in positions])
        
        if self.initial_position in self.walls:
            self.walls.remove(self.initial_position)
        if self.done_position in self.walls:
            self.walls.remove(self.done_position)
            
        for x in self.walls:
            if x in self._observation_space:
                self._observation_space.remove(x)
                
        for id, coord in enumerate(self._observation_space):
            self._states_mapping[coord] = id

        self.transition_probabilities = None
        self.rewards = None
        
        self._build_transition_probabilities()
        self._build_rewards()
            
    def _build_transition_probabilities(self):
        """Build the transition probabilit yfunction, of the form (state, action, next_state)
        """        
        S_dim = len(self.observation_space)
        self.transition_probabilities = np.zeros((S_dim, 4, S_dim))
        for s, ps in enumerate(self._observation_space):
            ps = Coordinate(*ps)
            if ps == self.done_position:
                self.transition_probabilities[s, Action.UP.value, s] = 1
                self.transition_probabilities[s, Action.LEFT.value, s] = 1
                self.transition_probabilities[s, Action.RIGHT.value, s] = 1
                self.transition_probabilities[s, Action.DOWN.value, s] = 1
                continue
                
            for s_next, ps_next in enumerate(self._observation_space):
                
                ps_next = Coordinate(*ps_next)
                
                if ps.is_close(ps_next) is False: continue
                
                
                
                #Prob: (1-slippery_probs) if action correct
                # slippery_probability * 0.5 * 0.5 if action incorrect
                # UP
                p_wrong = self.slippery_probability * (1 - self.stay_probability_after_slipping) * 0.5
                if ps.UP == ps_next:
                    self.transition_probabilities[s, Action.UP.value, s_next] += 1 - self.slippery_probability
                    self.transition_probabilities[s, Action.LEFT.value, s_next] += p_wrong
                    self.transition_probabilities[s, Action.RIGHT.value, s_next] += p_wrong
                elif ps.LEFT == ps_next:
                    self.transition_probabilities[s, Action.LEFT.value, s_next] += 1 - self.slippery_probability
                    self.transition_probabilities[s, Action.UP.value, s_next] += p_wrong
                    self.transition_probabilities[s, Action.DOWN.value, s_next] += p_wrong
                    
                elif ps.RIGHT == ps_next:
                    self.transition_probabilities[s, Action.RIGHT.value, s_next] += 1 - self.slippery_probability
                    self.transition_probabilities[s, Action.UP.value, s_next] += p_wrong
                    self.transition_probabilities[s, Action.DOWN.value, s_next] += p_wrong
                elif ps.DOWN == ps_next:
                    self.transition_probabilities[s, Action.DOWN.value, s_next] += 1 - self.slippery_probability
                    self.transition_probabilities[s, Action.LEFT.value, s_next] += p_wrong
                    self.transition_probabilities[s, Action.RIGHT.value, s_next] += p_wrong
                elif ps == ps_next:
                    self.transition_probabilities[s, Action.UP.value, s_next] += self.slippery_probability * self.stay_probability_after_slipping
                    self.transition_probabilities[s, Action.LEFT.value, s_next] += self.slippery_probability * self.stay_probability_after_slipping
                    self.transition_probabilities[s, Action.DOWN.value, s_next] += self.slippery_probability * self.stay_probability_after_slipping
                    self.transition_probabilities[s, Action.RIGHT.value, s_next] += self.slippery_probability * self.stay_probability_after_slipping
                    
                    # does not account for walls or borders!!
                    if ps.x == 0 or ps.LEFT in self.walls:
                        self.transition_probabilities[s, Action.UP.value, s_next] += p_wrong
                        self.transition_probabilities[s, Action.DOWN.value, s_next] += p_wrong
                        self.transition_probabilities[s, Action.LEFT.value, s_next] += (1-self.slippery_probability)
                    if ps.y == 0 or ps.DOWN in self.walls:
                        self.transition_probabilities[s, Action.LEFT.value, s_next] += p_wrong
                        self.transition_probabilities[s, Action.RIGHT.value, s_next] += p_wrong
                        self.transition_probabilities[s, Action.DOWN.value, s_next] += (1-self.slippery_probability)
                    if ps.y == self.n_rows - 1 or ps.UP in self.walls:
                        self.transition_probabilities[s, Action.LEFT.value, s_next] += p_wrong
                        self.transition_probabilities[s, Action.RIGHT.value, s_next] += p_wrong
                        self.transition_probabilities[s, Action.UP.value, s_next] += (1-self.slippery_probability)
                    if ps.x == self.n_columns - 1 or ps.RIGHT in self.walls:
                        self.transition_probabilities[s, Action.UP.value, s_next] += p_wrong
                        self.transition_probabilities[s, Action.DOWN.value, s_next] += p_wrong
                        self.transition_probabilities[s, Action.RIGHT.value, s_next] += (1-self.slippery_probability)
                        
                    
                    
                else:
                    raise Exception('There is a state that seems reachable but no action seems to reach it')
        
        self.transition_probabilities = self.transition_probabilities / self.transition_probabilities.sum(-1)[:,:,np.newaxis]
        # for i in range(4):
        #     self.transition_probabilities[:,i,:] = self.transition_probabilities[:, i, :] / self.transition_probabilities[:,i,:].sum(-1)[:, np.newaxis]
    
    def _build_rewards(self):
        """
        Build the reward functin of the form (state, action, next_state)
        """        
        S_dim = len(self.observation_space)
        self.rewards = np.zeros((S_dim, 4, S_dim))
        
        # NB We are assuming that these positions are not occupied by walls!
        pos1 = self.done_position.copy().LEFT
        pos2 = self.done_position.copy().DOWN
        
        self.rewards[
            self._states_mapping[pos1],
            Action.RIGHT.value,
            self._states_mapping[self.done_position]
        ] = 1
        
        self.rewards[
            self._states_mapping[pos1],
            Action.LEFT.value,
            self._states_mapping[self.done_position]
        ] = 1
        
        
        self.rewards[
            self._states_mapping[pos1],
            Action.DOWN.value,
            self._states_mapping[self.done_position]
        ] = 1
        
        
        self.rewards[
            self._states_mapping[pos2],
            Action.UP.value,
            self._states_mapping[self.done_position]
        ] = 1
        
        self.rewards[
            self._states_mapping[pos2],
            Action.RIGHT.value,
            self._states_mapping[self.done_position]
        ] = 1
        
        self.rewards[
            self._states_mapping[pos2],
            Action.LEFT.value,
            self._states_mapping[self.done_position]
        ] = 1


    @property
    def observation_space(self):
        return self._observation_space

    def reset(self) -> int:
        self.current_position = self.initial_position
        return self._states_mapping[self.current_position]

    def step(self, action: Action) -> Tuple[int, float, bool]:
        pos = self.current_position
        done = False
        reward = 0
        
        if self.current_position == self.done_position:
            # Absorbing state
            done = True
        else:
            if np.random.uniform() < self.slippery_probability:
                if np.random.uniform() < self.stay_probability_after_slipping:
                    return self._states_mapping[self.current_position], reward, done
                elif action == Action.UP or action == Action.DOWN:
                    action = Action.LEFT if np.random.uniform() < 0.5 else Action.RIGHT
                else:
                    action = Action.UP if np.random.uniform() < 0.5 else Action.DOWN

            match action:
                case Action.UP:
                    if pos.y != self.n_rows - 1 and pos.UP not in self.walls:
                        self.current_position = pos.UP
                case Action.RIGHT:
                    if pos.x != self.n_columns - 1 and pos.RIGHT not in self.walls:
                        self.current_position = pos.RIGHT
                case Action.DOWN:
                    if pos.y != 0 and  pos.DOWN not in self.walls:
                        self.current_position = pos.DOWN
                case Action.LEFT:
                    if pos.x != 0 and pos.LEFT not in self.walls:
                        self.current_position = pos.LEFT
                case _:
                    raise ValueError(f'Invalid action {action}')

            if self.current_position == self.done_position:
                reward = 1
                done = True
        
        return self._states_mapping[self.current_position], reward, done
    
    def show(self, policy: Optional[NDArray[np.float64]] = None):
        c = 65

        def _format_cell(pos, current_position, walls, done_position):
            if pos in walls:
                return 'X'
            
            if policy is None:
                if pos == current_position:
                    return 'P'
                elif pos == done_position:
                    return 'E'
                return ' '
            else:
                match Action(policy[self._states_mapping[pos]]):
                    case Action.UP: return '↑'
                    case Action.DOWN: return '↓'
                    case Action.LEFT: return '←'
                    case Action.RIGHT: return '→'

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

