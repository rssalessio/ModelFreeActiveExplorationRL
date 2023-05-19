# Model Free Active Exploration in Reinforcement Learning

This repository hosts the code accompanying the paper "A Model-Free Exploration Strategy for Reinforcement Learning". Our study approaches the exploration problem in Reinforcement Learning (RL) from an information-theoretical viewpoint and presents a novel, efficient, and entirely model-free solution.

*License*: MIT

*Additional license info*: the CartPoleSwingUp environment, in `CartPoleSwingUp/env`, is licensed under the Apache License 2.0.

## Requirements

To run the examples you need the following libraries: `numpy scipy cvxpy mosek torch matplotlib notebook`. 

## How to run the numerical results

### Example 4.3

All the code to run the simulations in Example 4.3 of the manuscript can be found in the `BoundsAnalysis` folder. To run the simulations, simply execute the jupyter notebook `BoundsAnalysis\run_example.ipynb`.  Make sure to create a `data` folder and a `figures` folder before running the simulations.

### Results for the tabular case


### Results for continuous MDPs