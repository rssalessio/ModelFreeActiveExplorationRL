import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def print_heatmap(env, states_visits, file_name, vmax):
    rows_labels = list(range(env.n_rows))
    rows_labels.reverse()

    visits_matrix = np.zeros((env.n_rows,env.n_columns))
    for state, visits in enumerate(states_visits):
        coord = list(env._states_mapping.keys())[list(env._states_mapping.values()).index(state)]
        coord = list(coord)
        coord[0] = env.n_rows-1-coord[0]
        visits_matrix[coord[0]][coord[1]] = visits
    
    fig, ax = plt.subplots()
    im = ax.imshow(visits_matrix, cmap='hot',vmin = 0, vmax = vmax)
    ax.set_yticks(np.arange(len(visits_matrix)), labels=rows_labels)
    import re
    plot_title = re.split('\.|/', file_name)[1]
    ax.set_title(plot_title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(file_name)
    plt.close()