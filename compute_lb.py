import numpy as np
import matplotlib.pyplot as plt
import argparse
from maze import Maze, MazeParameters, Action
from tqdm import tqdm
from empirical_model import EmpiricalModel
from policy_iteration import policy_iteration, policy_evaluation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from agent import QlearningAgent, Experience, GenerativeExplorativeAgent, Agent, Eq6Agent, OnPolicyAgent
from utils import print_heatmap, plot_results
from typing import Callable, Tuple
from BestPolicyIdentificationMDP.characteristic_time import CharacteristicTime, \
    compute_generative_characteristic_time, compute_characteristic_time_fw
import pickle
import os
DISCOUNT_FACTOR = 0.99
MAZE_PARAMETERS = MazeParameters(
    num_rows=16,
    num_columns=16,
    slippery_probability=0.3,
    walls=[(1,1), (2,2), (0,4), (1,4),  (4,0), (4,1), (4,4), (4,5), (4,6), (5,4), (5, 5), (5, 6), (6,4), (6, 5), (6, 6)],
    random_walls=False
)
NUM_EPISODES = 5000
NUM_ACTIONS = len(Action)
ACTIONS = list(Action)


def train() -> EmpiricalModel:
    env = Maze(MAZE_PARAMETERS)
    model = EmpiricalModel(len(env.observation_space), 4)
    for _ in tqdm(range(NUM_EPISODES)):
        state = env.reset()
        
        while True:
            action = np.random.choice(NUM_ACTIONS)
            next_state, reward, done = env.step(ACTIONS[action])
            model.update_visits(state, action, next_state, reward)
            state = next_state
            if done:
                break
        
    return model

if __name__ == '__main__':
    model = train()
    print("Computing generative lb")
    allocation_generative = compute_generative_characteristic_time(DISCOUNT_FACTOR, 
            model.transition_function, model.reward)
    
    print('Computing lb with constraints')
    allocation_with_constraints = compute_characteristic_time_fw(
        DISCOUNT_FACTOR, model.transition_function, model.reward,
        with_navigation_constraints=True, use_pgd=False, max_iter=3000
    )
    with open('data/lb_generative.pkl', 'wb') as f:
        pickle.dump(allocation_generative, f)

    with open('data/lb_with_constraints.pkl', 'wb') as f:
        pickle.dump(allocation_with_constraints, f)