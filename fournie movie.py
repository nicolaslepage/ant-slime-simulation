# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:34:35 2021

@author: Windows
"""

from fourmie2 import *
from numba import jit 
import matplotlib.pyplot as mpl
import pygame as pg
import random as rd
import numpy as np

import cv2

import glob

from scipy.ndimage.filters import uniform_filter, gaussian_filter


@jit(nopython=True)
def difuse(window, rate3, blurred):
    
    window = window*(1-rate3) + rate3*blurred
    
    return window
                
            
@jit(nopython=True)
def evap(window,rate) :
    return window*rate

def init_agent(n):
    
    agent_list1 = []
    agent_list2 = []
    agent_list3 = []
    
    for i in range (n) :
        agent_list1.append(slime_agent(np.shape(window),param1,param2,param3,0))
        agent_list2.append(slime_agent(np.shape(window),param1,param2,param3,-1))
        agent_list3.append(slime_agent(np.shape(window),param1,param2,param3,1))
    
    return agent_list1, agent_list2 , agent_list3


@jit(nopython=True)
def rounding(window):
    
    frame = np.zeros(np.shape(window), dtype=np.uint8)
    frame = np.round(window, 0, frame)

    return frame

@jit(nopython=True)
def rand_param():
    
    rate5 = rd.randint(1,2)
    rate3 = 0.75 + 0.25*rd.random()
    rate2 = 1 - (0.05*rd.random())
    param1 = 0.5 + 0.5*rd.random()
    param2 = 0.05*rd.random()
    param3 = rd.randint(1,7)

    return rate5, rate3, rate2, param1, param2, param3




######## init of windows #################

window = np.zeros((1000,1000,3),dtype=np.float64)

######## model parameters init #################

color = 255

last = 0

rate5, rate3, rate2, param1, param2, param3 = rand_param()

print( " gaussian blur = " + str(rate5) + "  blur multiplier = " + str(rate3) + "  evap rate = " + str(rate2) +  "  following param = " + str(param1) + "  rand threshold = " + str(param2) + "  speed = " + str(param3) )

out = cv2.VideoWriter('project1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 60, (np.shape(window)[0], np.shape(window)[1]))

######## model parameters init #################

steps = 15000

diverge = False

######## init of agents  #################

agent_list1, agent_list2 , agent_list3 = init_agent(5000)
    
######## game loop  #################
    
for i in range (steps) :
    
    last_diverge = diverge
    
    if i % 100 == 0 :
        print("step " + str(i) + " out of " + str(steps))
    
    change_param = False
    
    last_diverge = diverge
       
       
    if rd.random()<0.005:

        
        rate5, rate3, rate2, param1, param2, param3 = rand_param()
        
        change_param = True

        print( " gaussian blur = " + str(rate5) + "  blur multiplier = " + str(rate3) + "  evap rate = " + str(rate2) +  "  following param = " + str(param1) + "  rand threshold = " + str(param2) + "  speed = " + str(param3) )
        
    blurred = np.copy(window)
    
    blurred[:,:,0] = gaussian_filter(window[:,:,0], rate5, truncate=1)
    blurred[:,:,1] = gaussian_filter(window[:,:,0], rate5, truncate=1)
    blurred[:,:,2] = gaussian_filter(window[:,:,0], rate5, truncate=1)

    window = difuse(window, rate3, blurred)
    
    window = evap(window,rate2)
    
    if rd.random() < 0.002 :
        
        last = i
        
        diverge = diverge == False
         
        print("color changed")
        
    
    for i in range (len(agent_list1)) :
        
        if change_param :
            
            agent_list1[i].change_param( param1, param2, param3)
            agent_list2[i].change_param( param1, param2, param3)
            agent_list3[i].change_param( param1, param2, param3)
            
        
        pos1 = agent_list1[i].step(window[:,:,0])
        pos2 = agent_list2[i].step(window[:,:,1])
        pos3 = agent_list3[i].step(window[:,:,2])
                
        if diverge :  
            
            window[int(pos1[0,0]),int(pos1[0,1]),0] = color
            window[int(pos2[0,0]),int(pos2[0,1]),1] = color
            window[int(pos3[0,0]),int(pos3[0,1]),2] = color
            
            if last_diverge != diverge :
                
                agent_list1[i].sensiblock = 1
                agent_list2[i].sensiblock = 1
                agent_list3[i].sensiblock = 1  
                
        else :
            
            window[int(pos1[0,0]),int(pos1[0,1]),:] = [200, 200, 200]
            window[int(pos2[0,0]),int(pos2[0,1]),:] = [200, 200, 200]
            window[int(pos3[0,0]),int(pos3[0,1]),:] = [200, 200, 200]
            
    frame = rounding(window)

    out.write(frame)

out.release()




    
   
        