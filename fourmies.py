# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 20:18:56 2021

@author: Windows
"""

import matplotlib as mpl
import numpy as np
import random as rd
from numba import jit
from numba import jitclass
from numba import int32, float32    # import the types

# spec = [('location', int32[:,:]),('speed', float32[:,:]),('window', int32[:,:]),('sensiblock', int32),('sensor_positions', int32),('nudging_strenght', int32)]

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
        self.location[0,0] = rd.randint(self.maxspeed+1,window_size[0]-self.maxspeed-1)
        self.location[0,1] = rd.randint(self.maxspeed+1,window_size[0]-self.maxspeed-1)
        
        self.speed = np.zeros((1,2),dtype=np.int32)
        

        if initspeed == 0 :
            self.speed[0,0] = rd.randint(-self.maxspeed,self.maxspeed)
            self.speed[0,1] = rd.randint(-self.maxspeed,self.maxspeed)
        if initspeed == -1 :
            self.speed = self.location - self.window/2
            self.speed = 15*self.speed/(self.window[0,0]/2)
            self.speed = np.around(self.speed)
        if initspeed == 1 :
            self.speed = self.location - self.window/2
            self.speed = -15*self.speed/(self.window[0,0]/2)
            self.speed = np.around(self.speed)
        
        self.rand = random
        
        
    
    def step(self,field):
    
        nextspot = self.location + self.speed
        
        while nextspot[0,0] < self.maxspeed or nextspot[0,0] > self.window[0,0] - self.maxspeed or nextspot[0,1] < self.maxspeed or nextspot[0,1] > self.window[0,1] - self.maxspeed:
            
            self.speed[0,0] = rd.randint(-self.maxspeed,self.maxspeed)
            self.speed[0,1] = rd.randint(-self.maxspeed,self.maxspeed)
            nextspot = self.location + self.speed

        self.location = nextspot
        
        self.nudge2(field)
        
        return self.location
    
    def change_param(self,nudging, random, maxspeed):
        
        self.nudging_strenght = nudging
        self.rand = random
        self.maxspeed = maxspeed
    
          
    def nudge2(self,field):
        
        # print("initial speed = " + str(self.speed))
        # print("position is = " + str(self.location))
        
        # check inactivated sensibilisation
        
        if self.sensiblock <= 0 :
            
            # random action
            
            if rd.random() < self.rand :
                self.sensiblock = 5
                self.speed[0,0] = rd.randint(-self.maxspeed ,self.maxspeed )
                self.speed[0,1] = rd.randint(-self.maxspeed ,self.maxspeed )
            else :
                
                # creating look ahead points to know where to turn
                
                leftrotation = np.array([[np.sqrt(2)/2,-np.sqrt(2)/2],[np.sqrt(2)/2,np.sqrt(2)/2]])
                rightrotation = np.array([[np.sqrt(2)/2,np.sqrt(2)/2],[-np.sqrt(2)/2,np.sqrt(2)/2]])
                
                if np.absolute(self.speed[0,0]) < self.maxspeed and np.absolute(self.speed[0,1]) < self.maxspeed :
                    
                    right = np.zeros((1,2))
                    right= np.round(np.dot(self.speed*2,rightrotation))
                    # print("right is =" +str(right))
                    
                    
                    center= self.speed*2
                    # print("center is =" +str(center))
                    
                    
                    left = np.zeros((1,2))
                    left = np.round(np.dot(self.speed*2,leftrotation))
                    # print("left is =" +str(left))
                    
                else :
                    right = np.zeros((1,2))
                    right= np.round(np.dot(self.speed,rightrotation))
                    # print("right is =" +str(right))
                    
                    
                    center= self.speed
                    # print("center is =" +str(center))
                    
                    
                    left = np.zeros((1,2))
                    left = np.round(np.dot(self.speed,leftrotation))
                    # print("left is =" +str(left))
                    
                
                look=np.zeros((3,2))
                
                look[0,0] = self.location[0,0] + left[0,0] 
                look[0,1] = self.location[0,1] + left[0,1]
                
                look[1,0] = self.location[0,0] + center[0,0]
                look[1,1] = self.location[0,1] + center[0,1]
                
                look[2,0] = self.location[0,0] + right[0,0] 
                look[2,1] = self.location[0,1] + right[0,1] 
                
                newspeed=np.zeros((3,2))
                
                newspeed[0,0] = left[0,0] 
                newspeed[0,1] = left[0,1]
                
                newspeed[1,0] = center[0,0]
                newspeed[1,1] = center[0,1]
                
                newspeed[2,0] = right[0,0] 
                newspeed[2,1] = right[0,1] 

                
                argmax = 1
                maxval = -1
                
                for i in range(3):
                    
                    if look[i,0] < self.window[0,0] and look[i,0] > 0 and look[i,1] < self.window[0,1] and look[i,1] > 0 :
                        value = field[int(look[i,0]), int(look[i,1])]
                        if value > maxval and value>=1:
                            maxval = value
                            argmax = i    
                    
                # print("argmax=" +str(argmax))
                # print("nudge = " + str(self.nudging_strenght*newspeed[argmax]))     
                             
                self.speed = np.round(self.speed*(1-self.nudging_strenght) + self.nudging_strenght*newspeed[argmax,:])
                
                # print("newspeed =" + str(self.speed))
                # print("")
                
        else :
            self.sensiblock -= 1