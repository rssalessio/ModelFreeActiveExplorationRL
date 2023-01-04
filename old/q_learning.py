import numpy as np
import matplotlib.pyplot as plt
from envs.maze import Maze, MazeParameters, Action
from tqdm import tqdm
from empirical_model import EmpiricalModel
from policy_iteration import policy_iteration, policy_evaluation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from agent import QlearningAgent, Experience
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

agent = QlearningAgent(len(env.observation_space), NUM_ACTIONS, DISCOUNT_FACTOR, ALPHA)

FREQ_EVAL_GREEDY = 5

def print_heatmap(env, states_visits, episode):
    rows_labels = list(range(env.n_rows))
    rows_labels.reverse()

    visits_matrix = np.zeros((env.n_rows,env.n_columns))
    for state, visits in enumerate(states_visits):
        coord = list(env._states_mapping.keys())[list(env._states_mapping.values()).index(state)]
        coord = list(coord)
        coord[0] = env.n_rows-1-coord[0]
        visits_matrix[coord[0]][coord[1]] = visits
    
    fig, ax = plt.subplots()
    im = ax.imshow(visits_matrix, cmap='hot',vmin = 0, vmax = 700)
    ax.set_yticks(np.arange(len(visits_matrix)), labels=rows_labels)
    ax.set_title("State frequency episode {}".format(episode))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig("state_frequencies_dqn/State_frequency_episode_{}.pdf".format(episode))


for episode in tqdm(range(NUM_EPISODES)):
    state = env.reset()
    steps = 0
    rewards = 0
    if episode in [10, 20, 50, 70, 99]:
        print_heatmap(env, agent.num_visits_state, episode)
    while True:
        action = agent.forward(state, steps)
        next_state, reward, done = env.step(ACTIONS[action])
        model.update_visits(state, action, next_state, reward)
        agent.backward(Experience(state, action, reward, next_state, done))

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
            action = agent.greedy_action(state)#pi[state]
            next_state, reward, done = env.step(ACTIONS[action])
            state = next_state
            rewards_eval += reward
            if done:
                state = env.reset()
        print(f'[EVAL - {episode}] Total reward {rewards_eval}')


q_policy = agent.q_function.reshape(-1,4).argmax(1)
import pdb
#pdb.set_trace()
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
