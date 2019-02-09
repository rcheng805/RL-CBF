import tensorflow as tf
import numpy as np
from utils import *


#Generalized Advantage Estimator
class GAE():
    def __init__(self, sess, input_size, gamma, lamda, vf_constraint):
        self.sess = sess
        self.input_size = input_size
        self.gamma = gamma
        self.lamda = lamda
        self.vf_constraint = vf_constraint
        self.build_model()

    # Input will be state : [batch_size, observation size]
    # Outputs state value function
    def build_model(self):
        print('Initializing Value function network')
        with tf.variable_scope('VF'):
            self.x = tf.placeholder(tf.float32, [None, self.input_size], name='State')

            #Target will be the observation of the value function
            self.target = tf.placeholder(tf.float32, [None,1], name='Target')

            #Model is MLP composed of 3 hidden layers with 100, 50, 25 tanh units
            h1 = LINEAR(self.x, 100, name='h1')
            h1_n1 = tf.tanh(h1)
            h2 = LINEAR(h1_n1, 50, name='h2')
            h2_n1 = tf.tanh(h2)
            h3 = LINEAR(h2_n1, 25, name='h3')
            h3_n1 = tf.tanh(h3)
            self.value = LINEAR(h3_n1, 1, name='FC')

        tr_vrbs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='VF')
        for i in tr_vrbs:
            print(i.op.name)

        # Compute the loss, and gradient of loss w.r.t. neural network weights
        self.loss = tf.reduce_mean(tf.pow(self.target - self.value, 2))
        self.grad_objective = FLAT_GRAD(self.loss, tr_vrbs)

        self.y = tf.placeholder(tf.float32, [None])
        self.HVP = HESSIAN_VECTOR_PRODUCT(self.loss, tr_vrbs, self.y)

        #To adjust weights and biases
        self.get_value = GetValue(self.sess, tr_vrbs, name='VF')
        self.set_value = SetValue(self.sess, tr_vrbs, name='VF')

        self.sess.run(tf.global_variables_initializer())

    def get_advantage(self, paths):
        for path in paths:
            path["return_sum"] = DISCOUNT_SUM(path["Reward"], self.gamma)
        
        #Get observation, make it [batch_size, observation_size]
        self.observation = np.squeeze(np.concatenate([path["Observation"] for path in paths]))
        self.return_sum = np.concatenate([path["return_sum"] for path in paths])
        self.rewards = np.concatenate([path["Reward"] for path in paths])
        self.done = np.concatenate([path["Done"] for path in paths])
        
        #Get returns for each state (observation) to use for training value function
        batch_s = self.observation.shape[0]
        self.return_sum = np.resize(self.return_sum, [batch_s, 1])
        
        #Compute value function for all timesteps using current parameter
        feed_dict = {self.x:self.observation, self.target:self.return_sum}
        self.value_s = self.sess.run(self.value, feed_dict=feed_dict)
        self.value_s = np.resize(self.value_s, (batch_s,))
        
        #Next value function (if current state is before game done, set value as 0)
        self.value_next_s = np.zeros((batch_s,))
        self.value_next_s[:batch_s-1] = self.value_s[1:]
        self.value_next_s *= (1 - self.done)
        
        #delt_t_V : reward_t + gamma*Value(state_{t+1}) - Value(state_t)
        self.delta_v = np.squeeze(self.rewards) + self.gamma*self.value_next_s - self.value_s
        
        #Compute advantage estimator for all timesteps
        GAE = DISCOUNT_SUM(self.delta_v, self.gamma*self.lamda)
        
        #Normalize to make mean 0
        GAE = (GAE - np.mean(GAE)) / (np.std(GAE) + 1e-6)
        return GAE
    
    def train(self):
        #print('Training Value function network')
        #Get the parameter values for gradient, etc...
        parameter_prev = self.get_value()
        feed_dict = {self.x:self.observation, self.target:self.return_sum}
        gradient_objective = self.sess.run(self.grad_objective, feed_dict = feed_dict)
        
        #Function which takes 'y' input returns Hy
        def get_hessian_vector_product(y):
            feed_dict[self.y] = y
            return self.sess.run(self.HVP, feed_dict=feed_dict)
        
        def loss(parameter):
            self.set_value(parameter)
            return self.sess.run(self.loss, feed_dict=feed_dict)
        
        #Move theta in direction that minimizes loss (improve value function estimation)
        step_direction = CONJUGATE_GRADIENT(get_hessian_vector_product, -gradient_objective)
        
        #Determine step to satisfy constraint & minimize loss
        constraint_approx = 0.5*step_direction.dot(get_hessian_vector_product(step_direction))
        maximal_step_length = np.sqrt(self.vf_constraint / constraint_approx)
        full_step = maximal_step_length*step_direction
        
        #Estimation improvement, and set parameter to decrease loss
        new_parameter = LINE_SEARCH(loss, parameter_prev, full_step, name='Value loss')
        
        #Update without line search
        #new_parameter = parameter_prev + full_step
        
        #Set new parameter value
        self.set_value(new_parameter, update_info=0)
        
        
        