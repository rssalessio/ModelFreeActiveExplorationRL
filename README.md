# Model Free Active Exploration in Reinforcement Learning

This repository hosts the code accompanying the paper "A Model-Free Exploration Strategy for Reinforcement Learning", published at NeurIPS 2023. Our study approaches the exploration problem in Reinforcement Learning (RL) from an information-theoretical viewpoint and presents a novel, efficient, and entirely model-free solution.

**Authors**: Alessio Russo, Alexandre Proutiere \
**Code author**: Alessio Russo\
**License**: MIT\
**Additional license info**: the CartPoleSwingUp environment, in `CartPoleSwingUp/env`, was originally taken from  the BSuite repository (DeepMind Technologies Limited). The files in that directory were originally licensed under the APACHE 2.0 license, which can be found at the root of this repository. Changes and additions to those files are licensed under MIT.

## Requirements

To run the examples you need atleast Python 3.10 and  the following libraries installed: `numpy scipy cvxpy mosek torch matplotlib notebook tqdm seaborn pandas cython`. Additional libraries may be needed.

We also recommend to install the `MOSEK` solver.

## How to run the simulations

### Example 4.3

All the code to run the simulations in Example 4.3 of the manuscript can be found in the `BoundsAnalysis` folder. To run the simulations, simply execute the jupyter notebook `BoundsAnalysis\run_example.ipynb`.  Make sure to create a `data` folder and a `figures` folder before running the simulations.

### Results for the tabular case

Simulations for the tabular case can be found in the `RiverSwim` folder. To run the simulations, simply execute the file `run.py` (make sure to adjust the parameters, like `NUM_PROCESSES`, etc...). Use the `make_plots.ipynb` to plot the results.

### Results for continuous MDPs

1. To run the simulations for the `Slipping DeepSea` environment, execute the file `DeepSea\run_sim.py` (make sure to adjust the parameters `NUM_PROCESSES`, `N_SIMS`, `FREQ_EVAL_GREEDY`, etc...).Use the `make_plots.ipynb` to plot the results.
2. To run the simulations for the `Cartpole swingup` environment, execute the file `CartPoleSwingup\run_sim.py` (make sure to adjust the parameters inside the file).Use the `make_plots.ipynb` to plot the results. Other plots cna be done using the `make_entropy_plots.ipynb` and `make_exploration_pltos.ipynb` notebooks.

### Additional results

- To run the example of the 2-states MDP in the appendix, execute the jupyter notebook in `example_2statesMDP\run_example.ipynb`.

## FAQ/Problems

- If you encounter problems plotting the results using a Jupyter notebook on Linux, remember to install the necessary latex packages `sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super` (check here [https://stackoverflow.com/questions/11354149/python-unable-to-render-tex-in-matplotlib])
