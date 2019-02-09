import numpy as np
import cbf
import dynamics_gp
from barrier_comp import BARRIER

class LEARNER():
    def __init__(self,env, sess):
        self.firstIter = 1
        self.count = 1
        self.env = env
        self.torque_bound = 100.

        '''
        #Set up observation space and action space
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print('Observation space', self.observation_space)
        print('Action space', self.action_space)
        '''
        
        #Determine dimensions of observation & action space
        self.observation_size = 15
        self.action_size = 1
        
        # Build barrier function model
        cbf.build_barrier(self)
        
        # Build GP model of dynamics
        dynamics_gp.build_GP_model(self)
        
        self.bar_comp = BARRIER(sess,15,1)


