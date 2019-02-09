import tensorflow as tf
import numpy as np

#Initialize or run neural network layer
def LINEAR(x, hidden, name='None'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        #Set zero for initial weights (i.e. start out with 'empty' NN)
        #weight = tf.get_variable('Weight', [x.get_shape()[-1], hidden], initializer=tf.constant_initializer(0))
        weight = tf.get_variable('Weight', [x.get_shape()[-1], hidden], initializer=tf.truncated_normal_initializer(stddev=0.03))  # Near empty NN
        
        #Base initializer
        #weight = tf.get_variable('Weight', [x.get_shape()[-1], hidden], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('Bias', [hidden,], initializer=tf.constant_initializer(0))
        weighted_sum = tf.matmul(x,weight) + bias
    return weighted_sum

#Calculate probability of action (mu, logstd : [batch size, action size])
def LOG_POLICY(mu, logstd, action):
    #Get log probability of action given mean/variance
    variance = tf.exp(2*logstd)
    log_prob = -tf.square(action-mu)/(2*variance) - 0.5*tf.log(2*np.pi) - logstd

    #Sum along 'action size' axis to get product of probability of each action index
    return tf.reduce_sum(log_prob,1)

#KL divergence between two parameterized Gaussians
def GAUSS_KL(mu1, logstd1, mu2, logstd2):
    variance1 = tf.exp(2*logstd1)
    variance2 = tf.exp(2*logstd2)
    
    kl = logstd2 - logstd1 + (variance1 + tf.square(mu1 - mu2))/(2*variance2) - 0.5
    return tf.reduce_sum(kl)

#Shannon entropy of gaussian
def GAUSS_ENTROPY(mu, logstd):
    variance = tf.exp(2*logstd)
    entropy = (1 + tf.log(2*np.pi*variance))/2
    return tf.reduce_sum(entropy)

#CHECK THIS FUNCTION
#Flatten gradient along all variables
def FLAT_GRAD(loss, vrbs):
    # tf.gradients returns list of gradients w.r.t. variables
    '''
        If loss argument is list, tf.gradients returns sum of gradient of each loss element for each variable
        tf.gradients([y,z]) => [dy/dx + dz/dx] where x is variable
    '''
    grads = tf.gradients(loss, vrbs)

    #Ensures each gradient has same shape with variables    
    return tf.concat(values=[tf.reshape(g, [np.prod(v.get_shape().as_list()),]) for (g,v) in zip(grads,vrbs)], axis=0)

def GAUSS_KL_FIRST_FIX(mu, logstd):
    #First argument is old policy, so keep it unchanged through tf.stop_gradient
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd
    return GAUSS_KL(mu1, logstd1, mu2, logstd2)

# Get actual parameter values
class GetValue:
    def __init__(self, sess, variable_list, name=None):
        self.name = name
        self.sess = sess
        self.op_list = tf.concat(axis=0, values=[tf.reshape(v, [np.prod(v.get_shape().as_list())]) for v in variable_list])
        
    #Use class instance as function
    def __call__ (self):
        #print('Getting %s parameter value' % self.name)
        return self.op_list.eval(session=self.sess)
    
# Set policy parameter values
class SetValue:
    def __init__(self, sess, variable_list, name=None):
        self.name = name
        self.sess = sess
        shape_list = list()
        
        # Get 'shape/size' of variable list
        for i in variable_list:
            shape_list.append(i.get_shape().as_list())
        total_variable_size = np.sum(np.prod(shapes) for shapes in shape_list)
        print('Total variable size : %d' % total_variable_size)
        
        #Assign variables in variable list
        self.var_list = var_list = tf.placeholder(tf.float32,[total_variable_size])
        start = 0
        assign_ops = list()
        for (shape, var) in zip(shape_list, variable_list):
            variable_size = np.prod(shape)
            assign_ops.append(tf.assign(var, tf.reshape(var_list[start:(start+variable_size)], shape)))
            start += variable_size
        
        self.op_list = tf.group(*assign_ops)
        
    def __call__(self, var, update_info=0):
        if update_info:
            print('Update %s parameter' % self.name)
        self.sess.run(self.op_list, feed_dict={self.var_list:var})

# Take Hessian product with y (w.r.t. variables vrbs)
def HESSIAN_VECTOR_PRODUCT(func, vrbs, y):
    first_derivative = tf.gradients(func, vrbs)
    flat_y = list()
    start = 0
    for var in vrbs:
        variable_size = np.prod(var.get_shape().as_list())
        param = tf.reshape(y[start:(start+variable_size)], var.get_shape())
        flat_y.append(param)
        start += variable_size
    #First derivative * y
    gradient_with_y = [tf.reduce_sum(f_d * f_y) for (f_d, f_y) in zip(first_derivative, flat_y)]
    HVP = FLAT_GRAD(gradient_with_y, vrbs)
    return HVP

'''
    'x' should be array of shape [batch size, ]
    Return: [x1 + df*x2 + x3*df**2 + ..., x2 + df*x3 + (df**2)*x4 + ..., x3 + ..., ...]
'''
def DISCOUNT_SUM(x, discount_factor, print_info=None):
    size = x.shape[0]
    if print_info is not None:
        print('Input shape', size, 'Discount factor', discount_factor)
    discount_sum = np.zeros((size,))
    # x[::-1] is reverse of x
    for idx, value in enumerate(x[::-1]):
        discount_sum[:size-idx] += value
        if size-idx-1 == 0:
            break
        discount_sum[:size-idx-1] *= discount_factor
        
    return discount_sum
    
def CONJUGATE_GRADIENT(fvp, y, k=10, tolerance=1e-6):
    #Given initial guess, r0 := y-fvp(x0) but our initial value is x := 0 so r0 := y
    p = y.copy()
    r = y.copy()
    x = np.zeros_like(y)
    r_transpose_r = r.dot(r)
    for i in range(k):
        FIM_p = fvp(p)
        # alpha := r.t*r /p.t*A*p
        alpha_k = r_transpose_r / p.dot(FIM_p)
        
        #x_{k+1} = x_k + alpha_k*p
        x += alpha_k*p
        
        #r_{k+1} := r_k - alpha_k*A*p
        r -= alpha_k*FIM_p
        
        #beta_k = r_{k+1}.t*r_{k+1} / r_k.t*r_k
        new_r_transpose_r = r.dot(r)
        beta_k = new_r_transpose_r / r_transpose_r
        
        #p_{k+1} := r_{k+1} + beta_k*p_k
        p = r + beta_k*p
        r_transpose_r = new_r_transpose_r
        if r_transpose_r < tolerance:
            break
    
    return x
    
def LINE_SEARCH(surr, theta_prev, full_step, num_backtracking=10, name=None):
    #Get previous surrogate value (loss for Value function or reward for Policy)
    prev_sur_objective = surr(theta_prev)
    # backtracking :1,1/2,1/4,1/8
    for num_bt, fraction in enumerate(0.5**np.arange(num_backtracking)):
        #Exponentially shrink beta (step size)
        step_frac = full_step*fraction
        # theta -> theta + step
        theta_new = theta_prev + step_frac
        new_sur_objective = surr(theta_new)
        sur_improvement = prev_sur_objective - new_sur_objective
        if sur_improvement > 0:
            #print('%s improved from %3.4f to %3.4f' % (name, prev_sur_objective, new_sur_objective))
            return theta_new
    #print('Objective not improved - reverting to previous theta')
    return theta_prev
    
