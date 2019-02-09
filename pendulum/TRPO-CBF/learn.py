import tensorflow as tf
import numpy as np
from trpo import TRPO
from scipy.io import savemat
import gym
import datetime
from gym import spaces

class LEARNER():
    def __init__(self, args, sess, simulator):
        self.args = args
        self.sess = sess
        self.simulator = simulator

        #Construct simulation environment
        self.simulator = gym.make('Pendulum-v0')
        self.simulator.unwrapped.max_torque = 15.
        self.simulator.unwrapped.max_speed = 60.
        self.simulator.unwrapped.action_space = spaces.Box(low=-self.simulator.unwrapped.max_torque, high=self.simulator.unwrapped.max_torque, shape=(1,))
        high = np.array([1., 1., self.simulator.unwrapped.max_speed])
        self.simulator.unwrapped.observation_space = spaces.Box(low=-high, high=high)

        
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
            if train_index%5 == 0:
                self.agent.sim()
                print(train_index)

            if total_steps > self.args.total_train_step:
                savemat('data4_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',dict(data=all_logs, args=self.args))
                break
