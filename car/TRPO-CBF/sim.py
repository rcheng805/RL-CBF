#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:52:36 2018

@author: rcheng
"""

import tensorflow as tf
from car_simulator import Car, allCars
from scipy.io import savemat
import numpy as np
from trpo import TRPO
from learn import LEARNER
import argparse

#Parse arguments for TRPO learner
parser = argparse.ArgumentParser()
parser.add_argument('--num_backtracking', type=int, default=16)
parser.add_argument('--kl_constraint', type=float, default=0.0003)  #Originally 0.01
parser.add_argument('--timesteps_per_batch', type=int, default=1200)
parser.add_argument('--gamma', type=float, default=0.995)
parser.add_argument('--lamda', type=float, default=0.98)
parser.add_argument('--bar_constraint_max', type=float, default=1e1)
parser.add_argument('--vf_constraint', type=float, default=1e-2)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--max_path_length', type=int, default=80)
parser.add_argument('--total_train_step', type=int, default=1.2e6)
args = parser.parse_args()

#Set up tensorflow to use GPU
config = tf.ConfigProto()
config.log_device_placement = False
config.gpu_options.allow_growth = True

# Initialize environment for all cars
car_env = allCars()

#Run tensorflow session
with tf.Session(config=config) as sess:
    trpo_gae_learner = LEARNER(args,sess,car_env)
    trpo_gae_learner.learn()
    print("Learning Process Finished")
