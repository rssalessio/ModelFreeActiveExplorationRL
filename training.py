import numpy as np
import matplotlib.pyplot as plt
import argparse
import multiprocessing as mp
import pickle
from maze import Maze, MazeParameters, Action
from tqdm import tqdm
from empirical_model import EmpiricalModel
from policy_iteration import policy_iteration, policy_evaluation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from agent import QlearningAgent, Experience, GenerativeExplorativeAgent, Agent, Eq6Agent, OnPolicyAgent
from utils import print_heatmap, plot_results
from typing import Callable, Tuple


NUM_PROCESSES = 2
NUM_RUNS = 5
DISCOUNT_FACTOR = 0.99
MAZE_PARAMETERS = MazeParameters(
    num_rows=16,
    num_columns=16,
    slippery_probability=0.3,
    walls=[(1,1), (2,2), (0,4), (1,4),  (4,0), (4,1), (4,4), (4,5), (4,6), (5,4), (5, 5), (5, 6), (6,4), (6, 5), (6, 6)],
    random_walls=False
)
FREQ_EVAL_GREEDY = 300
NUM_EPISODES = 100
MAX_ITERATIONS = 20000
NUM_ACTIONS = len(Action)
ALPHA = 0.6
ACTIONS = list(Action)


def eval(env: Maze, agent: Agent) -> Tuple[float, float]:
    rewards_eval = []
    steps_eval = []
    for g_ep in range(10):
        rewards_eval_ep = 0
        steps_eval_ep = 0
        state = env.reset()
        for i in range(300):
            action = agent.greedy_action(state)
            next_state, reward, done = env.step(ACTIONS[action])
            state = next_state
            rewards_eval_ep += reward
            steps_eval_ep += 1
            if done:
                break
        rewards_eval.append(rewards_eval_ep)
        steps_eval.append(steps_eval_ep)
    return np.mean(rewards_eval), np.mean(steps_eval)

def train(method: str, id_run: int = 0):
    np.random.seed(id_run)
    env, greedy_env = Maze(MAZE_PARAMETERS), Maze(MAZE_PARAMETERS)
    model = EmpiricalModel(len(env.observation_space), 4)
    episode_rewards = []
    episode_steps = []
    greedy_rewards = []
    greedy_steps = []
    #env.show()

    make_agent, dir = create_agent_callable(method)
    agent = make_agent(env)

    iteration = 0
    
    results = {
        'num_visits_state': [],
        'last_visit_state': [],
        'policy_diff_generative': [],
        'policy_diff_constraints': [],
        'episode_rewards': [],
        'episode_steps': [],
        'greedy_rewards': [],
        'greedy_steps': []
    }

    #for episode in range(NUM_EPISODES):
    episode = 0
    while iteration < MAX_ITERATIONS:
        state = env.reset()
        steps = 0
        rewards = 0
        
        
        while True and iteration < MAX_ITERATIONS:
            if iteration % FREQ_EVAL_GREEDY == 0:
                grew, gsteps = eval(greedy_env, agent)
                greedy_rewards.append((iteration, grew))
                greedy_steps.append((iteration, gsteps))
                print(f'[EVAL - Episode:{episode} - iteration {iteration} - id {id_run}] Total reward {grew} - Total steps {gsteps}')
            
            
            if iteration % 500 == 0 and iteration > 0:
                results['num_visits_state'].append((iteration, agent.num_visits_state))
                results['last_visit_state'].append((iteration, agent.last_visit_state))
                
                
                # plot_results(env, agent,
                #              episode_rewards, episode_steps, greedy_rewards, greedy_steps,
                #              file_name=f'episode_{episode}_steps_{iteration}', dir=dir)
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
        
        
        episode_rewards.append((episode, rewards))
        episode_steps.append((episode, steps))
        episode += 1
        

    results['episode_rewards'] = episode_rewards
    results['episode_steps'] = episode_steps
    results['greedy_rewards'] = greedy_rewards
    results['greedy_steps'] = greedy_steps
    results['policy_diff_generative'] = agent.policy_diff_generative
    results['policy_diff_constraints'] = agent.policy_diff_constraints
    return results
    
    # V, pi, Q = policy_iteration(DISCOUNT_FACTOR, model.transition_function, model.reward)

    # print('Policy iteration')
    # env.show(pi)


def create_agent_callable(type: str) -> Tuple[Callable[[Maze], Agent], str]:
    match type:
        case 'generative':
            return lambda env: GenerativeExplorativeAgent(len(env.observation_space), NUM_ACTIONS, DISCOUNT_FACTOR), 'results_generative'
        case 'generative_with_constraints':
            return lambda env: GenerativeExplorativeAgent(len(env.observation_space), NUM_ACTIONS, DISCOUNT_FACTOR, navigation_constraints=True, frequency_computation=100), 'results_generative_with_constraints'
        case 'qlearning':
            return lambda env: QlearningAgent(len(env.observation_space), NUM_ACTIONS, DISCOUNT_FACTOR, ALPHA), 'results_qlearning'
        case 'eq6_model_based':
            return lambda env: Eq6Agent(len(env.observation_space), NUM_ACTIONS, DISCOUNT_FACTOR, ALPHA), 'results_eq6_model_based'
        case 'eq6_model_free':
            return lambda env: Eq6Agent(len(env.observation_space), NUM_ACTIONS, DISCOUNT_FACTOR, ALPHA, estimate_var=True), 'results_eq6_model_free'
        case 'onpolicy':
            return lambda env: OnPolicyAgent(len(env.observation_space), NUM_ACTIONS, DISCOUNT_FACTOR, lr=1e-2, hidden=16, training_period=64, alpha=0.6), 'results_onpolicy'
            
    return None, None

if __name__ == '__main__':
    # Usage python.py method_name
    parser = argparse.ArgumentParser()
    parser.add_argument("method", help="Choose between one of the methods",
                        type=str, default='generative', nargs='?', 
                        choices=['generative', 'generative_with_constraints', 'qlearning', 'eq6_model_based', 'eq6_model_free', 'onpolicy'])
    args = parser.parse_args()
    print(f'Method chosen: {args.method}')
    
    with mp.Pool(NUM_PROCESSES) as pool:
        results = pool.starmap(train, [(args.method, id_run) for id_run in range(NUM_RUNS)])

    with open(f'results_{args.method}.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)