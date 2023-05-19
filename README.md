# Model Free Active Exploration in Reinforcement Learning

This repository hosts the code accompanying the paper "A Model-Free Exploration Strategy for Reinforcement Learning". Our study approaches the exploration problem in Reinforcement Learning (RL) from an information-theoretical viewpoint and presents a novel, efficient, and entirely model-free solution.

*License*: MIT

*Additional license info*: the CartPoleSwingUp environment, in `CartPoleSwingUp/env`, was originally taken from  the BSuite repository (DeepMind Technologies Limited). The files in that directory were originally licensed under the APACHE 2.0 license, which can be found at the root of this repository. Changes and additions to those files are licensed under MIT.

## Requirements

To run the examples you need atleast Python 3.10 and  the following libraries installed: `numpy scipy cvxpy mosek torch matplotlib notebook tqdm seaborn pandas cython`. Additional libraries may be needed. 

We also recommend to install the `MOSEK` solver. 


## How to run the numerical results

### Example 4.3

All the code to run the simulations in Example 4.3 of the manuscript can be found in the `BoundsAnalysis` folder. To run the simulations, simply execute the jupyter notebook `BoundsAnalysis\run_example.ipynb`.  Make sure to create a `data` folder and a `figures` folder before running the simulations.

### Results for the tabular case


### Results for continuous MDPs


### Additional results

-  To run the example of the 2-states MDP in the appendix, execute the jupyter notebook in `example_2statesMDP\run_example.ipynb`.