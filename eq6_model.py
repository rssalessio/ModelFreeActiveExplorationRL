import numpy as np
import matplotlib.pyplot as plt
from maze import Maze, MazeParameters, Action
from tqdm import tqdm
from empirical_model import EmpiricalModel
from policy_iteration import policy_iteration, policy_evaluation
DISCOUNT_FACTOR = 0.99
MAZE_PARAMETERS = MazeParameters(
    num_rows=8,
    num_columns=8,
    slippery_probability=0.3,
    walls=[(1,1), (2,2), (0,4), (1,4),  (4,0), (4,1), (4,4), (4,5), (4,6), (5,4), (5, 5), (5, 6), (6,4), (6, 5), (6, 6)],
    random_walls=False
)
NUM_EPISODES = 35000
NUM_ACTIONS = len(Action)
ALPHA = 0.6
ACTIONS = list(Action)

env = Maze(MAZE_PARAMETERS)

num_visits_state = np.zeros(len(env.observation_space))
num_visits_actions = np.zeros((len(env.observation_space), NUM_ACTIONS))
q_function = np.zeros((len(env.observation_space), NUM_ACTIONS))

model = EmpiricalModel(len(env.observation_space), 4)

episode_rewards = []
episode_steps = []
env.show()

def compute_explorative_policy(state: int, model: EmpiricalModel, Q, V, pi, tol = 1e-3):
    avg_V = np.array([model.transition_function[state, a] @ V for a in range(NUM_ACTIONS)])
    var_V = np.array([model.transition_function[state, a] @ ((V - avg_V[a]) ** 2) for a in range(NUM_ACTIONS)])
    
    delta = np.array([[Q[s, pi[s]] - Q[s, a] for a in range (NUM_ACTIONS)] for s in range(len(env.observation_space))])
    delta_min = max(tol, np.min(delta))
    
    delta_sq = np.clip(delta[state], a_min=delta_min, a_max=None) ** 2
    rho = (1+ var_V) / delta_sq
    exp_policy = rho
    action_pi = pi[state]
    exp_policy[action_pi] = np.sqrt(rho[action_pi] * (rho.sum() - rho[action_pi]))
    
    exp_policy = exp_policy / exp_policy.sum()
    
    return exp_policy
    

FREQ_EVAL_GREEDY = 5

for episode in tqdm(range(NUM_EPISODES)):
    state = env.reset()
    steps = 0
    rewards = 0
    while True:
        num_visits_state[state] += 1

        if num_visits_state[state] > 2 * NUM_ACTIONS:
            V, pi, Q = policy_iteration(DISCOUNT_FACTOR, model.transition_function, model.reward)
            exp_policy = compute_explorative_policy(state, model, Q, V, pi)
        else:
            exp_policy = np.ones(NUM_ACTIONS) / NUM_ACTIONS
        action = np.random.choice(NUM_ACTIONS, p = exp_policy)
        
        num_visits_actions[state][action] += 1

        next_state, reward, done = env.step(ACTIONS[action])
        model.update_visits(state, action, next_state, reward)
        
        lr = 1 / (num_visits_actions[state][action] ** ALPHA)
        q_function[state][action] += lr * (reward + DISCOUNT_FACTOR * q_function[next_state].max() - q_function[state][action])

        state = next_state

        steps += 1
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
            action = q_function[state].argmax()#pi[state]
            next_state, reward, done = env.step(ACTIONS[action])
            state = next_state
            rewards_eval += reward
            if done:
                state = env.reset()
        print(f'[EVAL - {episode}] Total reward {rewards_eval}')

V, pi, Q = policy_iteration(DISCOUNT_FACTOR, model.transition_function, model.reward)

print('Policy iteration')
env.show(pi)
