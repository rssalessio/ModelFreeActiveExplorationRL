# Model Free Active Exploration in Reinforcement Learning

This repository hosts the code accompanying the paper "A Model-Free Exploration Strategy for Reinforcement Learning". Our study approaches the exploration problem in Reinforcement Learning (RL) from an information-theoretical viewpoint and presents a novel, efficient, and entirely model-free solution.

*License*: MIT

*Additional license info*: the CartPoleSwingUp environment, in `CartPoleSwingUp/env`, was originally taken from  the BSuite repository (DeepMind Technologies Limited). The files in that directory were originally licensed under the APACHE 2.0 license, which can be found at the root of this repository. Changes and additions to those files are licensed under MIT.


# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#

## Requirements

To run the examples you need the following libraries: `numpy scipy cvxpy mosek torch matplotlib notebook`. 

## How to run the numerical results

### Example 4.3

All the code to run the simulations in Example 4.3 of the manuscript can be found in the `BoundsAnalysis` folder. To run the simulations, simply execute the jupyter notebook `BoundsAnalysis\run_example.ipynb`.  Make sure to create a `data` folder and a `figures` folder before running the simulations.

### Results for the tabular case


### Results for continuous MDPs