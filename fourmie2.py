# -*- coding: utf-8 -*-
"""
Created on Sun May  2 12:49:09 2021

@author: Windows
"""

import numpy as np
import random as rd
from numba import jit
from numba import jitclass
from numba import int32, float32    # import the types


@jit(nopython=True)
def randspeed(maxspeed):
    
    speed =np.zeros((1,2))
    
    speed[0,0] = rd.randint(-maxspeed,maxspeed)
    speed[0,1] = rd.randint(-maxspeed,maxspeed)
    
    return speed

@jit(nopython=True)
def jitround(array):
    
    return np.round(array, 0, array)
    

@jit(nopython=True)
def speed_init(initspeed, location, window, maxspeed):
    
    if initspeed == 0 :
            speed = randspeed(maxspeed)
            
    if initspeed == -1 :
        
        speed = location - window*0.5
        speed = maxspeed*speed*(1/(window[0,0]/2))
        speed = jitround(speed)
        
    if initspeed == 1 :
        speed = location - window*0.5
        speed = -maxspeed*speed/(1/(window[0,0]/2))
        speed = jitround(speed)
    
    return speed

@jit(nopython=True)
def gen_nextspot(location, speed, maxspeed, window) :
    
        nextspot = location + speed
        
        while nextspot[0,0] < maxspeed or nextspot[0,0] > window[0,0] - maxspeed or nextspot[0,1] < maxspeed or nextspot[0,1] > window[0,1] - maxspeed:

            speed = speed_init(0, location, window, maxspeed)
            nextspot = location + speed
            
        return speed, nextspot

@jit(nopython=True)
def gen_rotation(speed, maxspeed):
    
    leftrotation = np.array([[np.sqrt(2)/2,-np.sqrt(2)/2],[np.sqrt(2)/2,np.sqrt(2)/2]])
    rightrotation = np.array([[np.sqrt(2)/2,np.sqrt(2)/2],[-np.sqrt(2)/2,np.sqrt(2)/2]])
    
    if np.absolute(speed[0,0]) < maxspeed and np.absolute(speed[0,1]) < maxspeed :
        
        right = np.zeros((1,2))
        right= jitround(np.dot(speed*2,rightrotation))

        center= speed*2

        left = np.zeros((1,2))
        left = jitround(np.dot(speed*2,leftrotation))
        
    else :
        
        right = np.zeros((1,2))
        right= jitround(np.dot(speed,rightrotation))
    
        center= speed
        
        left = np.zeros((1,2))
        left = jitround(np.dot(speed,leftrotation))
        
    return right, center, left

@jit(nopython=True)
def gen_look(location, right, center ,left) :
    
    look=np.zeros((3,2), dtype=np.int32)
                
    look[0,0] = location[0,0] + left[0,0] 
    look[0,1] = location[0,1] + left[0,1]
    
    look[1,0] = location[0,0] + center[0,0]
    look[1,1] = location[0,1] + center[0,1]
    
    look[2,0] = location[0,0] + right[0,0] 
    look[2,1] = location[0,1] + right[0,1]
    
    return look

@jit(nopython=True)
def gen_newspeed(right, center ,left) :
    
    newspeed=np.zeros((3,2))
                
    newspeed[0,0] = left[0,0] 
    newspeed[0,1] = left[0,1]
    
    newspeed[1,0] = center[0,0]
    newspeed[1,1] = center[0,1]
    
    newspeed[2,0] = right[0,0] 
    newspeed[2,1] = right[0,1] 
    
    return newspeed

@jit(nopython=True)
def find_speed(window, look, field) :
    
    argmax = 1
    maxval = -1
    
    for i in range(3):
        
        if look[i,0] < window[0,0] and look[i,0] > 0 and look[i,1] < window[0,1] and look[i,1] > 0 :
            
            value = field[int(look[i,0]), int(look[i,1])]
            
            if value > maxval and value>=1:
                
                maxval = value
                argmax = i  
    
    return argmax
    

# spec = [('location', int32[:,:]),
#         ('speed', float32[:,:]),
#         ('window', int32[:,:]),
#         ('sensiblock', int32),
#         ('maxspeed', int32),
#         ('nudging_strenght', int32),
#         ('rand', int32)]

# @jitclass(spec)   

class slime_agent():
    
    def __init__(self,window_size, nudging_strenght, random, maxspeed, initspeed):
        
        self.maxspeed = maxspeed
        
        self.sensiblock = 0
        
        self.window = np.zeros((1,2),dtype=np.int32)
        self.window[0,0] = window_size[0]
        self.window[0,1] = window_size[1]

        
        self.nudging_strenght = nudging_strenght
        
        self.location = np.zeros((1,2),dtype=np.int32)
        self.location[0,0] = rd.randint(self.maxspeed + 1, self.window[0,0] - self.maxspeed-1)
        self.location[0,1] = rd.randint(self.maxspeed + 1, self.window[0,1] - self.maxspeed-1)
        
        self.speed = speed_init(initspeed, self.location, self.window, self.maxspeed)
        
        self.rand = random
        
        
    
    def step(self,field):
        
        self.speed, self.location = gen_nextspot(self.location, self.speed, self.maxspeed, self.window)
        
        self.nudge2(field)
        
        return self.location
    
    
    
    def change_param(self,nudging, random, maxspeed):
        
        self.nudging_strenght = nudging
        self.rand = random
        self.maxspeed = maxspeed
        
    
          
    def nudge2(self,field):
        
        # check inactivated sensibilisation
        
        if self.sensiblock <= 0 :
            
            # random action
            
            if rd.random() < self.rand :
                
                self.sensiblock = 5
                self.speed = speed_init(0, self.location, self.window, self.maxspeed)
                
            else :

                right, center, left = gen_rotation(self.speed, self.maxspeed)
                
                look = gen_look(self.location, right, center ,left)
                
                newspeed = gen_newspeed(right, center ,left)

                argmax = 1
                maxval = -1000
                    
                argmax = find_speed(self.window, look, field)    
                             
                self.speed = jitround(self.speed*(1-self.nudging_strenght) + self.nudging_strenght*newspeed[argmax,:])

        else :
            self.sensiblock -= 1