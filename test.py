import numpy as np
import matplotlib.pyplot as plt
from maze import Maze, MazeParameters, Action
from tqdm import tqdm
from empirical_model import EmpiricalModel
from policy_iteration import policy_iteration

DISCOUNT_FACTOR = 0.99
MAZE_PARAMETERS = MazeParameters(
    num_rows=8,
    num_columns=8,
    slippery_probability=0.3,
    walls=[(1,1), (2,2), (0,4), (1,4),  (4,0), (4,1), (4,4), (4,5), (4,6), (5,4), (5, 5), (5, 6), (6,4), (6, 5), (6, 6)],
    random_walls=False
)
NUM_EPISODES = 100
NUM_ACTIONS = len(Action)
ALPHA = 0.6
ACTIONS = list(Action)
DISCOUNT_FACTOR = 0.99
env = Maze(MAZE_PARAMETERS)
model = EmpiricalModel(MAZE_PARAMETERS.num_rows, MAZE_PARAMETERS.num_columns, 4)

V, pi = policy_iteration(DISCOUNT_FACTOR, model.transition_function, model.reward)
print(V)
print(pi)