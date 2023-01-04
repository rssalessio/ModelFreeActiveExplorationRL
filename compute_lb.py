import numpy as np
import pickle
from envs.maze import Maze, Action
from tqdm import tqdm
from empirical_model import EmpiricalModel
from BestPolicyIdentification import CharacteristicTime, \
    compute_characteristic_time, compute_generative_characteristic_time
from maze_parameters import MAZE_PARAMETERS, DISCOUNT_FACTOR

MIN_SAMPLING = 10000
NUM_ACTIONS = len(Action)
ACTIONS = list(Action)

def train() -> EmpiricalModel:
    env = Maze(MAZE_PARAMETERS)
    model = EmpiricalModel(len(env.observation_space), 4)
    frequencies = np.zeros((len(env.observation_space), 4))
    
    mask = np.ones((len(env.observation_space))).astype(bool)
    mask[env._states_mapping[env.done_position]] =  False     
    
    min_sampling = 0
    num_eps = 0
    
    with tqdm(total=MIN_SAMPLING) as pbar:
        state = env.reset()
        while min_sampling < MIN_SAMPLING:
            action = frequencies[state].argmin()
            frequencies[state][action] += 1
            next_state, reward, done = env.step(ACTIONS[action])
            model.update_visits(state, action, next_state, reward)
            state = next_state
            if done:
                num_eps += 1
                state = env.reset()
                delta = frequencies[mask].min() - min_sampling
                min_sampling += delta
                pbar.update(delta)
    print(f'Num epps {num_eps}')
    return model

if __name__ == '__main__':
    np.random.seed(0)
    model = train()
    print("Computing generative lb")
    allocation_generative = compute_generative_characteristic_time(DISCOUNT_FACTOR, 
            model.transition_function, model.reward)
    
    print('Computing lb with constraints')
    allocation_with_constraints = compute_characteristic_time(
        DISCOUNT_FACTOR, model.transition_function, model.reward,
        with_navigation_constraints=True, use_pgd=False, max_iter=3000
    )

    with open('data/lb_generative.pkl', 'wb') as f:
        pickle.dump(allocation_generative, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/lb_with_constraints.pkl', 'wb') as f:
        pickle.dump(allocation_with_constraints, f, protocol=pickle.HIGHEST_PROTOCOL)