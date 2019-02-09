#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 14:17:49 2018

@author: rcheng
"""

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def build_GP_model(self):
    N = 6  # 6 GPs for 6 states involved in CBF                                                      
    GP_list = []
    noise = 0.5
    for i in range(N):
        kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        gp = GaussianProcessRegressor(kernel=kern, alpha = noise, n_restarts_optimizer=10)
        GP_list.append(gp)
    self.GP_model = GP_list


#Build GP dynamics model
def update_GP_dynamics(self,path):
    N = self.observation_size
    X = path['Observation']
    U = path['Action']
    L = X.shape[0]
    err = np.zeros((L-1,N))
    for i in range(L-1):
        t = 0.05*(i % 80)
        [f,g,x1] = self.env.returnStep(X[i,:], t)
        f = np.ravel(f)
        g = np.ravel(g)
        x1 = np.ravel(x1)
        err[i,:] = X[i+1,:] - f - g*U[i]
    S = X[0:L-1,:]
    self.GP_model[0].fit(S,err[:,6])
    self.GP_model[1].fit(S,err[:,7])
    self.GP_model[2].fit(S,err[:,9])
    self.GP_model[3].fit(S,err[:,10])
    self.GP_model[4].fit(S,err[:,12])
    self.GP_model[5].fit(S,err[:,13])
    
def get_GP_dynamics(self, obs, u_rl, t):
    [f_nom,g,x] = self.env.returnStep(obs,t)
    f_nom = np.ravel(f_nom)
    g = np.ravel(g)
    x = np.ravel(x)
    f = np.copy(f_nom)
    [m1, std1] = self.GP_model[0].predict(x.reshape(1,-1), return_std=True)
    [m2, std2] = self.GP_model[1].predict(x.reshape(1,-1), return_std=True)
    [m3, std3] = self.GP_model[2].predict(x.reshape(1,-1), return_std=True)
    [m4, std4] = self.GP_model[3].predict(x.reshape(1,-1), return_std=True)
    [m5, std5] = self.GP_model[4].predict(x.reshape(1,-1), return_std=True)
    [m6, std6] = self.GP_model[5].predict(x.reshape(1,-1), return_std=True)
    f[6] = f_nom[6] + self.GP_model[0].predict(x.reshape(1,-1))
    f[7] = f_nom[7] + self.GP_model[1].predict(x.reshape(1,-1))
    f[9] = f_nom[9] + self.GP_model[2].predict(x.reshape(1,-1))
    f[10] = f_nom[10] + self.GP_model[3].predict(x.reshape(1,-1))
    f[12] = f_nom[12] + self.GP_model[4].predict(x.reshape(1,-1))
    f[13] = f_nom[13] + self.GP_model[5].predict(x.reshape(1,-1))
    return [np.squeeze(f), np.squeeze(g), np.squeeze(x), np.array([0,0,0,0,0,0,np.squeeze(std1),
                       np.squeeze(std2), 0, np.squeeze(std3), np.squeeze(std4), 0, np.squeeze(std5), np.squeeze(std6), 0])]

    
def get_GP_dynamics_prev(self, obs, u_rl, t):
    [f_nom,g,x] = self.env.returnStep(obs,t)
    f_nom = np.ravel(f_nom)
    g = np.ravel(g)
    x = np.ravel(x)
    f = f_nom
    [m1, std1] = self.GP_model_prev[0].predict(x.reshape(1,-1), return_std=True)
    [m2, std2] = self.GP_model_prev[1].predict(x.reshape(1,-1), return_std=True)
    [m3, std3] = self.GP_model_prev[2].predict(x.reshape(1,-1), return_std=True)
    [m4, std4] = self.GP_model_prev[3].predict(x.reshape(1,-1), return_std=True)
    [m5, std5] = self.GP_model_prev[4].predict(x.reshape(1,-1), return_std=True)
    [m6, std6] = self.GP_model_prev[5].predict(x.reshape(1,-1), return_std=True)
    f[6] = f_nom[6] + self.GP_model_prev[0].predict(x.reshape(1,-1))
    f[7] = f_nom[7] + self.GP_model_prev[1].predict(x.reshape(1,-1))
    f[9] = f_nom[9] + self.GP_model_prev[2].predict(x.reshape(1,-1))
    f[10] = f_nom[10] + self.GP_model_prev[3].predict(x.reshape(1,-1))
    f[12] = f_nom[12] + self.GP_model_prev[4].predict(x.reshape(1,-1))
    f[13] = f_nom[13] + self.GP_model_prev[5].predict(x.reshape(1,-1))
    return [np.squeeze(f), np.squeeze(g), np.squeeze(x), np.array([0,0,0,0,0,0,np.squeeze(std1),
                       np.squeeze(std2), 0, np.squeeze(std3), np.squeeze(std4), 0, np.squeeze(std5), np.squeeze(std6), 0])]
