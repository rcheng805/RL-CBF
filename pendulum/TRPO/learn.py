import tensorflow as tf
import numpy as np
from trpo import TRPO
from scipy.io import savemat
import datetime
import gym

class LEARNER():
    def __init__(self, args, sess, simulator):
        self.args = args
        self.sess = sess
        self.simulator = simulator

        #Construct simulation environment
        self.simulator = gym.make('Pendulum-v0')
        
        #Define learning agent (TRPO)
        self.agent = TRPO(self.args, self.simulator, self.sess)

    def learn(self):
        train_index = 0
        total_episode = 0
        total_steps = 0
        all_logs = list()
        while True:
            #Train the TRPO agent
            train_index += 1
            train_log = self.agent.train()
            total_steps += train_log["Total Step"]
            total_episode += train_log["Num episode"]

            all_logs.append(train_log)
            
            #Simulate system w/ new parameters
            if train_index%20 == 0:
                self.agent.sim()

            if total_steps > self.args.total_train_step:
                #nn_weights = {'policy_network': self.agent.get_value(), 'advantage_network': self.agent.gae.get_value()}
                savemat(self.args.name + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(data=all_logs, args=self.args))
                #savemat('weights_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(policy_weights=nn_weights, args=self.args))
                break
