import numpy as np
import matplotlib.pyplot as plt
from envs.maze import Maze, MazeParameters, Action
from tqdm import tqdm
from empirical_model import EmpiricalModel
from policy_iteration import policy_iteration, policy_evaluation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from agent import QlearningAgent, Experience, GenerativeExplorativeAgent
from utils import print_heatmap
import os
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

env = Maze(MAZE_PARAMETERS)


model = EmpiricalModel(len(env.observation_space), 4)

episode_rewards = []
episode_steps = []
env.show()

agent = GenerativeExplorativeAgent(len(env.observation_space), NUM_ACTIONS, DISCOUNT_FACTOR, navigation_constraints=True)

FREQ_EVAL_GREEDY = 5

iteration = 0
for episode in tqdm(range(NUM_EPISODES)):
    state = env.reset()
    steps = 0
    rewards = 0
    while True:
        if iteration in [500 * (i + 1) for i in range(10)]:
            print_heatmap(env, agent.num_visits_state,
                        f'State_frequency_episode_{episode}_steps_{iteration}',
                        dir='state_frequencies_generative_with_constraints')
            print_heatmap(env, agent.last_visit_state,
                        f'State_last_visit_episode_{episode}_steps_{iteration}',
                        dir='state_frequencies_generative_with_constraints')
        action = agent.forward(state, iteration)
        next_state, reward, done = env.step(ACTIONS[action])
        model.update_visits(state, action, next_state, reward)
        agent.backward(Experience(state, action, reward, next_state, done))

        state = next_state

        steps += 1
        iteration += 1
        rewards += reward
        if done:
            break
    
    episode_rewards.append(rewards)
    episode_steps.append(steps)
    
    if episode % FREQ_EVAL_GREEDY == 0:
        V, pi, Q = policy_iteration(DISCOUNT_FACTOR, model.transition_function, model.reward)
        state = env.reset()
        rewards_eval = 0
        for i in range(1000):
            action = agent.greedy_action(state)#pi[state]
            next_state, reward, done = env.step(ACTIONS[action])
            state = next_state
            rewards_eval += reward
            if done:
                state = env.reset()
        print(f'[EVAL - {episode}] Total reward {rewards_eval}')

omega = agent.allocation.omega
visits = agent.num_visits_actions



V, pi, Q = policy_iteration(DISCOUNT_FACTOR, model.transition_function, model.reward)

print('Policy iteration')
env.show(pi)

import pdb
pdb.set_trace()