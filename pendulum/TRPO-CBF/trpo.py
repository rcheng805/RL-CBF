import numpy as np
import tensorflow as tf
from utils import *
from gae import GAE
from barrier_comp import BARRIER
import cbf
import dynamics_gp

class TRPO():
    def __init__(self, args, env, sess):
        self.firstIter = 1
        self.count = 1
        self.args = args
        self.sess = sess
        self.env = env
        self.torque_bound = 15.
        self.max_speed = 60.

        #Set up observation space and action space
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        print('Observation space', self.observation_space)
        print('Action space', self.action_space)

        #Determine dimensions of observation & action space
        self.observation_size = self.env.observation_space.shape[0]
        self.action_size = self.action_space.shape[0]

        # Build neural network model for observations/actions
        self.build_model()
        
        # Build barrier function model
        cbf.build_barrier(self)
        
        # Build GP model of dynamics
        dynamics_gp.build_GP_model(self)


    # Build RL policy improvement model based on TRPO
    def build_model(self):
        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])
        self.action = tf.placeholder(tf.float32, [None, self.action_size])
        self.advantage = tf.placeholder(tf.float32, [None])

        #Mean of old action distribution
        self.old_action_dist_mu = tf.placeholder(tf.float32, [None, self.action_size])
        self.old_action_dist_logstd = tf.placeholder(tf.float32, [None, self.action_size])

        #NN framework for action distribution
        self.action_dist_mu, action_dist_logstd = self.build_policy(self.obs)

        # Get trainable variables for the policy (NN weights)                                                     
        tr_vrbs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Policy')
        for i in tr_vrbs:
            print(i.op.name)
        
        #Construct distribution by repeating action_dis_logstd
        self.action_dist_logstd = tf.tile(action_dist_logstd, (tf.shape(action_dist_logstd)[0],1))

        #Probability of action under old policy vs. new policy
        self.log_policy = LOG_POLICY(self.action_dist_mu, self.action_dist_logstd, self.action)
        self.log_old_policy = LOG_POLICY(self.old_action_dist_mu, self.old_action_dist_logstd, self.action)
        policy_ratio = tf.exp(self.log_policy - self.log_old_policy)
        
        #Number of observations in batch
        batch_size = tf.cast(tf.shape(self.obs)[0], tf.float32)
        
        '''
        Equation (14) in paper
        Contribution of a single s_n : Expectation over a~q[ (new policy / q(is)) * advantage_old]
        '''
        surr_single_state = -tf.reduce_mean(policy_ratio*self.advantage)
        
        
        #Define KL divergence and shannon entropy, averaged over a set of inputs (policies)        
        kl = GAUSS_KL(self.old_action_dist_mu, self.old_action_dist_logstd, self.action_dist_mu, self.action_dist_logstd) / batch_size
        ent = GAUSS_ENTROPY(self.action_dist_mu, self.action_dist_logstd) / batch_size
        
        
        #Define 'loss' quantities to constrain or maximize
        self.losses = [surr_single_state, kl, ent]
        
        # Maximize surrogate function over policy parameter 'theta' represented by neural network weights
        self.pg = FLAT_GRAD(surr_single_state, tr_vrbs)
        
        #KL divergence where first argument is fixed
        kl_first_fixed = GAUSS_KL_FIRST_FIX(self.action_dist_mu, self.action_dist_logstd) / batch_size
        
        #Gradient of KL divergence w.r.t. theta (NN policy weights)
        first_kl_grads = tf.gradients(kl_first_fixed, tr_vrbs)
        
        self.flat_tangent = tf.placeholder(tf.float32,[None])
        tangent = list()
        start = 0
        for vrbs in tr_vrbs:
            variable_size = np.prod(vrbs.get_shape().as_list())
            param = tf.reshape(self.flat_tangent[start:(start+variable_size)], vrbs.get_shape())
            tangent.append(param)
            start += variable_size
        '''
            Gradient of KL with tangent vector
            gradient_w_tangent : list of KL_prime*y for each variables
        '''
        gradient_w_tangent = [tf.reduce_sum(kl_g*t) for (kl_g, t) in zip(first_kl_grads, tangent)]
        
        '''
			From derivative of KL_prime*y : [dKL/dx1, dKL/dx2...]*y
				y -> Ay, A is n by n matrix but hard to implement(numerically solving (n*n)*(n*1))
				so first multiply target 'y' to gradient and take derivation
		    'self.FVP'	Returns : [d2KL/dx1dx1+d2KL/dx1dx2..., d2KL/dx1dx2+d2KL/dx2dx2..., ...]*y
			So get (second derivative of KL divergence)*y for each variable => y->JMJy (Fisher Vector Product)
		'''
        self.FVP = FLAT_GRAD(gradient_w_tangent, tr_vrbs)
        
        #Get actual parameter value
        self.get_value = GetValue(self.sess, tr_vrbs, name='Policy')
        
        #Set parameter values
        self.set_value = SetValue(self.sess, tr_vrbs, name='Policy')
        
        #Estimate of the advantage function 
        self.gae = GAE(self.sess, self.observation_size, self.args.gamma, self.args.lamda, self.args.vf_constraint)

        #Intialization of the barrier function compensator
        self.bar_comp = BARRIER(self.args, self.sess, self.observation_size, self.action_size)

        #Variable initializers
        self.sess.run(tf.global_variables_initializer())
        

    #Train TRPO policy 
    def train(self):
        batch_path = self.rollout()
        theta_prev = self.get_value()
        
        #Get advantage from gae (train value function NN)
        advantage_estimated = self.gae.get_advantage(batch_path)

        #Get barrier compensator from barrier_comp (train compensator NN)
        self.bar_comp.get_training_rollouts(batch_path)
        barr_loss = self.bar_comp.train()
        
        #Put all paths in batch in a numpy array to feed to network as [batch size, action/observation size]
        #Those batches come from OLD policy before updating theta
        action_dist_mu = np.squeeze(np.concatenate([each_path["Action_mu"] for each_path in batch_path]))
        action_dist_logstd = np.squeeze(np.concatenate([each_path["Action_logstd"] for each_path in batch_path]))
        observation = np.squeeze(np.concatenate([each_path["Observation"] for each_path in batch_path]))
        action = np.squeeze(np.concatenate([each_path["Action"] for each_path in batch_path]))

        
        
        #Obtain policy gradient of advantage function w.r.t. theta (g in paper)
        feed_dict = {self.obs:observation, self.action:np.expand_dims(action, axis=1), self.advantage:advantage_estimated, self.old_action_dist_mu:np.expand_dims(action_dist_mu, axis=1), self.old_action_dist_logstd:np.expand_dims(action_dist_logstd, axis=1)}
        #feed_dict = {self.obs:observation, self.action:action, self.advantage:advantage_estimated, self.old_action_dist_mu:action_dist_mu, self.old_action_dist_logstd:action_dist_logstd}
        policy_g = self.sess.run(self.pg, feed_dict=feed_dict)
        
        # Computing fisher vector product : FIM * (policy gradient) where FIM = Fisher Information Matrix
        def fisher_vector_product(gradient):
            feed_dict[self.flat_tangent] = gradient
            return self.sess.run(self.FVP, feed_dict=feed_dict)
            
        #Solve Ax = g, where A is FIM and g is gradient of policy network, to obtain search direction for theta
        search_direction = CONJUGATE_GRADIENT(fisher_vector_product, -policy_g)
        
        #KL divergence approximated by 1/2*(delta_transpose)*FIM*delta
        #Appendix C in TRPO Paper
        kl_approximated = 0.5*search_direction.dot(fisher_vector_product(search_direction))
        
        #Calculate theta update
        maximal_step_length = np.sqrt(self.args.kl_constraint / kl_approximated)
        full_step = maximal_step_length * search_direction
        
        def surrogate(theta):
            self.set_value(theta)
            return self.sess.run(self.losses[0], feed_dict=feed_dict)
            
        #Use line search to ensure improvement of surrogate objective and satisfaction of KL constraint
        #Start with maximal step length and exponentially shrink until objective improves
        new_theta = LINE_SEARCH(surrogate, theta_prev, full_step, self.args.num_backtracking, name='Surrogate loss')
        
        #Update without line search
        #new_theta = theta_prev + full_step
        
        #Update policy parameter theta
        self.set_value(new_theta, update_info=0)
        
        #Update value function neural network
        #Policy update is performed using old value function parameter
        self.gae.train()
        
        #After update, store values at log
        surrogate_after, kl_after, _ = self.sess.run(self.losses, feed_dict=feed_dict)
        logs = {"Surrogate loss":surrogate_after, "KL_DIV":kl_after}
        logs["Total Step"] = sum([len(path["Reward"]) for path in batch_path])
        logs["Num episode"] = len([path["Reward"] for path in batch_path])
        logs["Total Sum"] = sum([sum(path["Reward"]) for path in batch_path])
        logs["Episode Avg. Reward"] = logs["Total Sum"] / logs["Num episode"]
        logs["Compensator_Fit"] = barr_loss
        logs["Final_Action"] = np.squeeze(np.concatenate([each_path["Action"] for each_path in batch_path]))
        logs["Action_bar"] = np.squeeze(np.concatenate([each_path["Action_bar"] for each_path in batch_path]))
        logs["Action_BAR"] = np.squeeze(np.concatenate([each_path["Action_BAR"] for each_path in batch_path]))
        logs["Observation"] = np.squeeze(np.concatenate([each_path["Observation"] for each_path in batch_path]))
        logs["Reward"] = np.squeeze(np.concatenate([each_path["Reward"] for each_path in batch_path]))
        return logs
        
        
    #Set up NN to parameterize the control policy
    def build_policy(self, states, name='Policy'):
        print('Initializing Policy network')
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            h1 = LINEAR(states, self.args.hidden_size, name='h1')
            h1_n1 = tf.nn.relu(h1)
            h2 = LINEAR(h1_n1, self.args.hidden_size, name='h2')
            h2_n1 = tf.nn.relu(h2)
            h3 = LINEAR(h2_n1, self.action_size, name='h3')

            #Initialize action std_deviation
            #init = lambda shape, dtype, partition_info=None : 0.01*np.random.randn(*shape)
            #action_dist_logstd = tf.get_variable('logstd', initializer=init, shape=[1, self.action_size])
            
            #Initialize action std_deviation (no variance -- deterministic policy)
            action_dist_logstd = tf.get_variable('logstd', initializer=tf.constant_initializer(-1.), shape=[1, self.action_size])
            
        return h3, action_dist_logstd
    
    #Get action from the current observation (sampled based on NN policy)
    def act(self, obs):
        #Need to expand first dimension (batch axis), make [1, observation size]
        obs_expanded = np.expand_dims(np.squeeze(obs), 0)
        #obs_expanded = obs
        #Get action distribution from policy network
        action_dist_mu, action_dist_logstd = self.sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs:obs_expanded})
        #Sample action from gaussian distribution
        action = np.random.normal(loc=action_dist_mu, scale=np.exp(action_dist_logstd))
        return action, action_dist_mu, action_dist_logstd


    #Simulate dynamics for a given rollout
    def rollout(self):
        #Initialize variables
        paths = list()
        timesteps = 0
        self.num_epi = 0

        #Utilize GP from previous iteration while training current iteration
        if (self.firstIter == 1):
            pass
        else:
            self.GP_model_prev = self.GP_model.copy()
            dynamics_gp.build_GP_model(self)
        '''
        #Rebuild the GP model at the specified time interval
        if (self.count == 3):
            self.GP_model_prev = self.GP_model
            dynamics_gp.build_GP_model(self)
        '''
        
        #Iterate through the specified number of episodes
        while timesteps < self.args.timesteps_per_batch:
            self.num_epi += 1
            
            #Reset the environment
            obs, action, rewards, done, action_dist_mu, action_dist_logstd, action_bar, action_BAR = [], [], [], [], [], [], [], []
            prev_obs = self.env.reset()
            while (self.env.unwrapped.state[0] > 1 or self.env.unwrapped.state[0] < -1):
                prev_obs = self.env.reset()
            
            #Simulate dynamics for specified time
            for i in range(self.args.max_path_length):
                #self.env.render()
                prev_obs_expanded = np.expand_dims(np.squeeze(prev_obs), 0)
                #prev_obs_expanded = prev_obs
                #Agent takes actions from sampled action and action distribution parameters based on observation
                #All have shape of [1, action size]
                action_rl, action_dist_mu_rl, action_dist_logstd_ = self.act(prev_obs)

                #Utilize compensation barrier function
                u_BAR_ = self.bar_comp.get_action(prev_obs)
                action_RL = action_rl + u_BAR_
                action_dist_mu_RL = action_dist_mu_rl + u_BAR_

                #Utilize safety barrier function
                if (self.firstIter == 1):
                    [f,g,x,std] = dynamics_gp.get_GP_dynamics(self,prev_obs_expanded, action_RL)
                else:
                    [f,g,x,std] = dynamics_gp.get_GP_dynamics_prev(self,prev_obs_expanded, action_RL)
                u_bar_ = cbf.control_barrier(self, np.squeeze(prev_obs_expanded), action_dist_mu_RL, f, g, x, std)
                #action_ = action_RL + u_bar_
                action_dist_mu_ = action_dist_mu_RL + u_bar_

                #Stochastic action
                action_ = np.random.normal(loc=action_dist_mu_, scale=np.exp(action_dist_logstd_))
                
                #Store observation and action/distribution
                obs.append(prev_obs_expanded)
                action_bar.append(u_bar_)
                action_BAR.append(u_BAR_)
                action.append(action_)
                action_dist_mu.append(action_dist_mu_)
                action_dist_logstd.append(action_dist_logstd_)
                
                # Simulate dynamics after action
                next_obs, reward_, done_, _ = self.env.step(action_)
                
                #Get results
                done.append(done_)
                rewards.append(reward_)
                prev_obs = next_obs
                
                if done_:
                    path = {"Observation":np.concatenate(obs),
                            "Action":np.concatenate(action),
                            "Action_mu":np.concatenate(action_dist_mu),
                            "Action_bar":np.concatenate(action_bar),
                            "Action_BAR":np.concatenate(action_BAR),
                            "Action_logstd":np.concatenate(action_dist_logstd),
                            "Done":np.asarray(done),
                            "Reward":np.asarray(rewards)}
                    paths.append(path)
                    break
            #For timing purposes, only update GP dynamics for certain number of timesteps
            if (timesteps < 650):
                dynamics_gp.update_GP_dynamics(self,path)
            timesteps += len(rewards)
        #print('%d episodes, %d steps collected for batch' % (self.num_epi, timesteps))
        self.count = self.count + 1
        self.firstIter = 0
        return paths


    #Simulate/Visualize latest policy
    def sim(self):
        observation = self.env.reset()
        while (self.env.unwrapped.state[0] > 1 or self.env.unwrapped.state[0] < -1):
                observation = self.env.reset()

        total = 0
        for t in range(200):
            #Render environment
            self.env.render()
            
            #Get action from NN policy
            obs_expanded = np.expand_dims(np.squeeze(observation), 0)
            
            #Get action distribution from policy network
            action_dist_mu, action_dist_logstd = self.sess.run([self.action_dist_mu, self.action_dist_logstd], feed_dict={self.obs:obs_expanded})

            #Sample action from gaussian distribution
            action_rl = np.random.normal(loc=action_dist_mu, scale=np.exp(action_dist_logstd))

            #Get compensatory barrier action
            u_BAR_ = self.bar_comp.get_action(obs_expanded)
            u_RL = action_rl + u_BAR_
            
            #Compensate with barrier-based control
            [f,g,x,std] = dynamics_gp.get_GP_dynamics(self,obs_expanded,u_RL)
            u_bar = cbf.control_barrier(self, obs_expanded, u_RL, f, g, x, std)
            action = u_bar + u_RL
            

            
            observation, reward, done, info = self.env.step(action)
            total = total + reward
            if done:
                print("Accumulated Reward: {}".format(total))
                break                
