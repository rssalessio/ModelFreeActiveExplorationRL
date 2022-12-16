#!/bin/bash

rm -rf results_*

python training.py qlearning
python training.py generative
python training.py eq6_model_free
python training.py onpolicy
# python training.py generative_with_constraints

python training.py eq6_model_based
