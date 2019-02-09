import tensorflow as tf
import numpy as np
import argparse
from learn import LEARNER

#Suppress CPU warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():

    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_backtracking', type=int, default=20)
    parser.add_argument('--kl_constraint', type=float, default=0.005)  #Originally 0.01
    parser.add_argument('--timesteps_per_batch', type=int, default=3000)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--lamda', type=float, default=0.98)
    parser.add_argument('--bar_constraint_max', type=float, default=1e1)
    parser.add_argument('--vf_constraint', type=float, default=1e-2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--max_path_length', type=int, default=300)
    parser.add_argument('--total_train_step', type=int, default=1.8e6)
    args = parser.parse_args()
    
    #Set up tensorflow to use GPU
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True

    #Run tensorflow session
    with tf.Session(config=config) as sess:
        trpo_gae_learner = LEARNER(args,sess,args)
        trpo_gae_learner.learn()
        print("Learning Process Finished")

if __name__ == "__main__":
    main()
