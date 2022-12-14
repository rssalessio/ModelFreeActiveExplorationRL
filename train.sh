#!/bin/bash

rm -rf results_*

python training.py generative
python training.py generative_with_constraints
python training.py qlearning
python training.py eq6_model_based
python training.py eq6_model_free