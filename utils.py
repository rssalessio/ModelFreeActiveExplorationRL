import numpy as np
import matplotlib.pyplot as plt
import re
import os
from typing import List, Tuple
from mpl_toolkits.axes_grid1 import make_axes_locatable

def print_heatmap(env, states_visits: np.ndarray, file_name: str,  dir: str):
    rows_labels = list(range(env.n_rows))
    rows_labels.reverse()

    visits_matrix = np.zeros((env.n_rows,env.n_columns))
    for state, visits in enumerate(states_visits):
        coord = list(env._states_mapping.keys())[list(env._states_mapping.values()).index(state)]
        coord = list(coord)
        coord[0] = env.n_rows-1-coord[0]
        visits_matrix[coord[0]][coord[1]] = visits

    vmax = np.max(visits_matrix)
    fig, ax = plt.subplots()
    im = ax.imshow(visits_matrix, cmap='hot',vmin = 0, vmax = vmax)
    ax.set_yticks(np.arange(len(visits_matrix)), labels=rows_labels)
    
    plot_title = file_name #re.split('\.|/', file_name)[1]

    ax.set_title(plot_title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    if not os.path.exists(dir): os.makedirs(dir)
    plt.savefig(f'{dir}/{file_name}.pdf')
    plt.close()


def get_visits_matrix(env, states_visits):
    visits_matrix = np.zeros((env.n_rows,env.n_columns))
    for state, visits in enumerate(states_visits):
        coord = list(env._states_mapping.keys())[list(env._states_mapping.values()).index(state)]
        coord = list(coord)
        coord[0] = env.n_rows-1-coord[0]
        visits_matrix[coord[0]][coord[1]] = visits
    return visits_matrix

def plot_results(
        env,
        frequency_state_visits: np.ndarray,
        last_state_visits: np.ndarray,
        episode_rewards: List[Tuple[int, float]],
        episode_steps: List[Tuple[int, float]],
        greedy_rewards: List[Tuple[int, float]],
        greedy_steps: List[Tuple[int, float]],
        file_name: str,  dir: str):
    if not os.path.exists(dir): os.makedirs(dir)
    rows_labels = list(range(env.n_rows))
    rows_labels.reverse()
    
    fig, ax = plt.subplots(1, 2)
    
    # Plot frequency states
    freq_visits = get_visits_matrix(env, frequency_state_visits)
    vmax = np.max(freq_visits)
    im = ax[0].imshow(freq_visits, cmap='hot',vmin = 0, vmax = vmax)
    ax[0].set_yticks(np.arange(len(freq_visits)), labels=rows_labels)
    ax[0].set_title('Frequency of visitation')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    
    # Plot last visit
    last_visit = get_visits_matrix(env, last_state_visits)
    vmax = np.max(last_visit)
    im = ax[1].imshow(last_visit, cmap='hot',vmin = 0, vmax = vmax)
    ax[1].set_yticks(np.arange(len(last_visit)), labels=rows_labels)
    ax[1].set_title('State last visit')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)
    
    plt.suptitle(file_name)
    plt.savefig(f'{dir}/{file_name}_exploration.pdf')
    plt.close()
    
    fig, ax = plt.subplots(2, 2)
    
    # Plot episodes rewards
    if len(episode_rewards) > 0:
        episodes, rewards = zip(*episode_rewards)
        ax[0,0].plot(episodes, rewards)
        #ax[0,0].set_xlabel('Episode')
        ax[0,0].set_title('Total reward explorative')
    
    # Plot episodes steps
    if len(episode_steps) > 0:
        episodes, steps = zip(*episode_steps)
        ax[0,1].plot(episodes, steps)
        #ax[0,1].set_xlabel('Episode')
        ax[0,1].set_title('Total steps explorative')
    
    # Plot episodes rewards greedy
    if len(greedy_rewards) > 0:
        episodes, rewards = zip(*greedy_rewards)
        ax[1,0].plot(episodes, rewards)
        ax[1,0].set_xlabel('Episode')
        ax[1,0].set_title('Total  reward greedy')
    
    # Plot episodes steps
    if len(greedy_steps) > 0:
        episodes, steps = zip(*greedy_steps)
        ax[1,1].plot(episodes, steps)
        ax[1,1].set_xlabel('Episode')
        ax[1,1].set_title('Total steps greedy')
    
    plt.suptitle(file_name)
    plt.savefig(f'{dir}/{file_name}_rewards.pdf')
    plt.close()
    
    
