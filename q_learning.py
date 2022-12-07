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
NUM_EPISODES = 100
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


FREQ_EVAL_GREEDY = 5


for episode in tqdm(range(NUM_EPISODES)):
    state = env.reset()
    steps = 0
    rewards = 0
    while True:
        num_visits_state[state] += 1
        eps = 1 if num_visits_state[state] <= 2 * NUM_ACTIONS else max(0.5, 1 / (num_visits_state[state] - 2*NUM_ACTIONS))

        action = np.random.choice(NUM_ACTIONS) if np.random.uniform() < eps else q_function[state].argmax()
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


q_policy = q_function.reshape(-1,4).argmax(1)
import pdb
pdb.set_trace()
Vq = policy_evaluation(DISCOUNT_FACTOR, model.transition_function, model.reward, q_policy)

V, pi, Q = policy_iteration(DISCOUNT_FACTOR, model.transition_function, model.reward)

print('Policy iteration')
env.show(pi)

print('Q-learning')
env.show(q_policy)

# plt.plot(episode_steps)
# plt.title('Episode steps')
# plt.grid()
# plt.yscale('log')
# plt.show()
