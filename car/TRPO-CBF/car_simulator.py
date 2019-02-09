#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 14:40:23 2018

@author: rcheng
"""
import numpy as np

class Car():
    
    #Initialize car parameters
    def __init__(self,pos,vel,behindCar,frontCar):
        self.pos = pos
        self.vel = vel
        self.accel = 0
        self.behindCar = behindCar
        self.frontCar = frontCar
        self.u = 0
        self.dt = 0.05
        
    #Take next step for car, given action "act"
    def nextStep(self,act,act_noise):
        ''' 
        Simple double integrator dynamics for car
        doubledotx = a*u - b*dotx
        '''
        kd = 0.1  #Introduce damping uncertainty
        a = 1.   # Multipler for action --> acceleration
        action = np.random.normal(act,act_noise)
        accel = a*action - kd*self.vel
        self.vel = self.vel + accel*self.dt
        self.pos = 0.5*accel*self.dt**2 + self.vel*self.dt + self.pos
        self.accel = accel
    
    #Return state of car
    def getState(self):
        return [self.pos, self.vel, self.accel]
    
    #Update ordering of car
    def updateOrder(self,behindCar,frontCar):
        self.frontCar = frontCar
        self.behindCar = behindCar
        
    #Get next action of the car
    def getNextAction(self, vel_des):
        kp = 4
        k_brake = 20
        action = kp*(vel_des - self.vel)
        if self.frontCar:
            if (self.frontCar.pos - self.pos < 6.0):
                action = action - k_brake*(self.frontCar.pos - self.pos)

        return action
    
    #Get next action of last car
    def getNextAction5(self, vel_des):
        kp = 4
        k_brake = 20
        action = kp*(vel_des - self.vel)
        if self.frontCar:
            if (self.frontCar.frontCar.pos - self.pos < 12.0):
                action = action - k_brake*(self.frontCar.frontCar.pos - self.pos)*0.5

        return action

class allCars():
    
    #Initialize car environment
    def __init__(self):
        self.count = 0
        self.L = 80
        self.t = 0
        self.numCars = 5
        self.car_1 = Car(34.0,30.0,None,None)
        self.car_2 = Car(28.0,30.0,None,self.car_1)      
        self.car_3 = Car(22.0,30.0,None,self.car_2)       
        self.car_4 = Car(16.0,30.0,None,self.car_3)      
        self.car_5 = Car(10.0,30.0,None,self.car_4)   
        self.car_1.updateOrder(self.car_2,None)
        self.car_2.updateOrder(self.car_3,self.car_1)
        self.car_3.updateOrder(self.car_4,self.car_2)
        self.car_4.updateOrder(self.car_5,self.car_3)
        self.car_5.updateOrder(None,self.car_4)
        
        self.allStates = np.zeros((5,3))
        self.allStates = self.getAllStates()

    # Get states of all cars
    def getAllStates(self):
        self.allStates[0,:] = self.car_1.getState()
        self.allStates[1,:] = self.car_2.getState()
        self.allStates[2,:] = self.car_3.getState()
        self.allStates[3,:] = self.car_4.getState()
        self.allStates[4,:] = self.car_5.getState()
        return self.allStates
        
    # Get reward for all cars
    def getReward(self, action):
        s = self.car_4.getState()
        if (action > 0):
            r = -s[1]*action
        else:
            r = 0
        if (self.car_4.frontCar.pos - self.car_4.pos) < 2.99:
            r = r - np.abs(500/(self.car_4.frontCar.pos - self.car_4.pos))
            #print("Cars Close")
        if (self.car_4.pos - self.car_4.behindCar.pos) < 2.99:
            r = r - np.abs(500/(self.car_4.pos - self.car_4.behindCar.pos))
            #print("Cars Close")

        return r
    
    # Reset car environment
    def reset(self):
        self.t = 0
        self.numCars = 5
        self.car_1 = Car(34.0,30.0,None,None)             
        self.car_2 = Car(28.0,30.0,None,self.car_1)    
        self.car_3 = Car(22.0,30.0,None,self.car_2)      
        self.car_4 = Car(16.0,30.0,None,self.car_3)  
        self.car_5 = Car(10.0,30.0,None,self.car_4)   
        self.car_1.updateOrder(self.car_2,None)
        self.car_2.updateOrder(self.car_3,self.car_1)
        self.car_3.updateOrder(self.car_4,self.car_2)
        self.car_4.updateOrder(self.car_5,self.car_3)
        self.car_5.updateOrder(None,self.car_4)

        self.allStates = np.zeros((5,3))
        self.allStates = self.getAllStates()
        
        return np.ravel(self.allStates)
    
    #Project next step in car environment using nominal (incorrect) model
    def returnStep(self, obs,t):
        s = np.copy(obs)
        s = np.reshape(s,(5,3))
        x = np.copy(s)
        #Define dynamics
        dt = 0.05
        kp = 3.5
        kb = 18
        v_des = 30
        A = np.array([[0, 1],[kb, -kp]])
        
        #Car 5
        if (s[2,0] - s[4,0] < 12.0):
            A = np.array([[0, 1],[0.5*kb, -kp]])
            a1 = np.matmul(A,s[4,0:2])
            b = np.array([0, kp*v_des - 0.5*kb*s[2,0]])
        else:
            A = np.array([[0, 1],[0, -kp]])
            a1 = np.matmul(A,s[4,0:2])
            b = np.array([0, kp*v_des])
        a = a1 + b
        s[4,0] = s[4,0] + s[4,1]*dt + 0.5*a[1]*dt**2
        s[4,1] = s[4,1] + a[1]*dt
        s[4,2] = self.car_5.getNextAction(30)
        
        #Car 4 (current)
        s[3,0] = s[3,0] + s[3,1]*dt
        s[3,1] = s[3,1]
        
        #Car 3
        if (s[1,0] - s[2,0] < 6):
            A = np.array([[0, 1],[kb, -kp]])
            a1 = np.matmul(A,s[2,0:2])
            b = np.array([0, kp*v_des - kb*s[1,0]])
        else:
            A = np.array([[0, 1],[0, -kp]])
            a1 = np.matmul(A,s[2,0:2])
            b = np.array([0, kp*v_des])
        a = a1 + b
        s[2,0] = s[2,0] + s[2,1]*dt + 0.5*a[1]*dt**2
        s[2,1] = s[2,1] + a[1]*dt
        s[2,2] = self.car_3.getNextAction(30)
        
        #Car 2
        if (s[0,0] - s[1,0] < 6):
            A = np.array([[0, 1],[kb, -kp]])
            a1 = np.matmul(A,s[1,0:2])
            b = np.array([0, kp*v_des - kb*s[0,0]])
        else:
            A = np.array([[0, 1],[0, -kp]])
            a1 = np.matmul(A,s[1,0:2])
            b = np.array([0, kp*v_des])
        a = a1 + b
        s[1,0] = s[1,0] + s[1,1]*dt + 0.5*a[1]*dt**2
        s[1,1] = s[1,1] + a[1]*dt
        s[1,2] = self.car_2.getNextAction(30)
        
        #Car 1
        a = self.car_1.getNextAction(30-10*np.sin(0.2*t))
        s[0,0] = 0.5*a*dt**2 + s[0,1]*dt + s[0,0]
        s[0,1] = s[0,1] + a*dt
        s[0,2] = 0
        f = s
        
        #Actuated dynamics
        g = np.zeros((5,3))
        g[3,0] = 0.5*dt**2
        g[3,1] = dt
        g[3,2] = 0
        
        return f, g, x
    
    #Take step for all cars
    def step(self,action):
        #Take action for all cars
        self.car_5.nextStep(self.car_5.getNextAction5(30),5.0)
        self.car_4.nextStep(action,0.0)
        self.car_3.nextStep(self.car_3.getNextAction(30),5.0)
        self.car_2.nextStep(self.car_2.getNextAction(30),5.0)
        self.car_1.nextStep(self.car_1.getNextAction(30-10*np.sin(0.2*self.t)),5.0)
        r = self.getReward(action)
        s = np.ravel(self.getAllStates())
        self.t = self.t + 0.05
        self.count = self.count + 1
        if (self.count == self.L):
            self.count = 0
            return s, r, True
        else:
            return s, r, False
        
