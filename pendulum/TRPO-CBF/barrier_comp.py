import tensorflow as tf
import numpy as np
from utils import *
from sklearn.linear_model import LinearRegression

#Barrier Function Compensator
class BARRIER():
    def __init__(self, args, sess, input_size, action_size):
        self.sess = sess
        self.args = args
        self.input_size = input_size
        self.action_size = action_size
        self.build_model()

    # Input will be state : [batch_size, observation size]
    # Ouput will be control action
    def build_model(self):
        print('Initializing Barrier Compensation network')
        with tf.variable_scope('Compensator'):
            #Input will be observation
            self.x = tf.placeholder(tf.float32, [None, self.input_size], name='Obs')
            #Target will be control action
            self.target = tf.placeholder(tf.float32, [None, self.action_size], name='Target_comp')

            #Model is MLP composed of 2 hidden layers with 50, 40 relu units
            h1 = LINEAR(self.x, 50, name='h1')
            h1_n1 = tf.nn.relu(h1)
            h2 = LINEAR(h1_n1, 40, name='h2')
            h2_n1 = tf.nn.relu(h2)
            self.value = LINEAR(h2_n1, self.action_size, name='h3')

        tr_vrbs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Compensator')
        for i in tr_vrbs:
            print(i.op.name)

        # Compute the loss and gradient of loss w.r.t. neural network weights
        self.loss = tf.reduce_mean(tf.pow(self.target - self.value, 2))
        self.grad_objective = FLAT_GRAD(self.loss, tr_vrbs)

        self.y = tf.placeholder(tf.float32, [None])
        self.HVP = HESSIAN_VECTOR_PRODUCT(self.loss, tr_vrbs, self.y)

        #To adjust weights
        self.get_value = GetValue(self.sess, tr_vrbs, name='Compensator')
        self.set_value = SetValue(self.sess, tr_vrbs, name='Compensator')

        self.sess.run(tf.global_variables_initializer())

    def get_training_rollouts(self, paths):
        #Get observations and actions
        self.observation= np.squeeze(np.concatenate([path["Observation"] for path in paths]))
        self.action_bar = np.concatenate([path["Action_bar"] for path in paths])
        self.action_BAR = np.concatenate([path["Action_BAR"] for path in paths])

        #Reshape observations & actions to use for training
        batch_s = self.observation.shape[0]
        self.action_bar = np.resize(self.action_bar, [batch_s, self.action_size])
        self.action_BAR = np.resize(self.action_BAR, [batch_s, self.action_size])

    #Given current observation, get the neural network output (representing barrier compensator)
    def get_action(self, obs):
        observation = np.expand_dims(np.squeeze(obs),0)
        feed_dict = {self.x:observation}
        u_bar = self.sess.run(self.value, feed_dict)
        return u_bar
        
    def train(self):
        #print('Training barrier function compensator')
        for i in range(100):
            #Get the parameter values for gradient, etc...
            parameter_prev = self.get_value()
            action_comp = self.action_bar + self.action_BAR

            feed_dict = {self.x:self.observation, self.target:action_comp}
            gradient_objective = self.sess.run(self.grad_objective, feed_dict = feed_dict)
    
            #Function which takes 'y' input and returns Hy
            def get_hessian_vector_product(y):
                feed_dict[self.y] = y
                return self.sess.run(self.HVP, feed_dict=feed_dict)
    
            #Get loss under current parameter
            def loss(parameter):
                self.set_value(parameter)
                return self.sess.run(self.loss, feed_dict=feed_dict)
            '''
            #Move theta in direction that minimizes loss (improve barrier function parameterization)
            step_direction = CONJUGATE_GRADIENT(get_hessian_vector_product, -gradient_objective)
    
            #Determine step to satisfy contraint and minimize loss
            constraint_approx = 0.5*step_direction.dot(get_hessian_vector_product(step_direction))
            maximal_step_length = np.sqrt(self.args.bar_constraint_max /constraint_approx)
            full_step = maximal_step_length*step_direction
            '''
            
            full_step = -self.args.bar_constraint_max*gradient_objective
            #Set parameter to decrease loss - use line search to check
            new_parameter = LINE_SEARCH(loss, parameter_prev, full_step, name='Barrier loss')
            self.set_value(new_parameter, update_info=0)
            if (np.array_equal(new_parameter, parameter_prev)):
                print("Break")
                return loss(new_parameter)
                break
        return loss(new_parameter)
            #print("Compensator Loss: %3.4f" % loss(new_parameter))
