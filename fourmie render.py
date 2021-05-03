# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 00:12:49 2021

@author: Windows
"""

from fourmies import *
from numba import jit 
import matplotlib.pyplot as mpl
import pygame as pg

from scipy.ndimage.filters import gaussian_filter

@jit(nopython=True)
def difuse(window,size,rate):
    for i in range (size,np.shape(window)[0]-1-size):
        for j in range (size,np.shape(window)[1]-1-size):
            summ = 0.0
            for n in (0,2*size+1):
                for m in (0,2*size+1):
                    summ+= window[i+n-size,j+m-size]
                        
            for n in (0,2*size+1):
                for m in (0,2*size+1):
                    window[i+n-size,j+m-size] = window[i+n-size,j+m-size]*(1-rate) + (summ/9)*rate
    return window



def difuse_gauss(window,rate):
    return gaussian_filter(window, rate, truncate=1)
                
            
@jit(nopython=True)
def evap(window,rate) :
    return window*rate
    
    

window = np.zeros((1000,1000,3),dtype=np.float64)

agent_list = []

######## model parameters init #################

color = rd.randint(50,250)
rate2 = 1-(0.1*rd.random())
rate1 = rd.randint(1,2)
param1 = rd.random()
param2 = 0.1**rd.randint(1,3)
param3 = rd.randint(1,10)

for i in range (1000) :
    agent_list.append(slime_agent(np.shape(window),param1,param2,param3,0))
    
pg.init()

screen = pg.display.set_mode(np.shape(window))

running = True
    
surf = pg.surfarray.make_surface(window)

running = True


while running:
    
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
    
    change_param = False
    
    if rd.random()<0.01:
        color = rd.randint(50,250)
        rate2 = 1-(0.1*rd.random())
        rate1 = rd.randint(1,2)
        param1 = rd.random()
        param2 = 0.1**rd.randint(1,3)
        param3 = rd.randint(1,20)
        change_param = True
        print(str(color)+" "+str(rate2)+" "+str(rate1)+" "+str(param1)+" "+str(param2)+" "+str(param3))
    
    window=difuse_gauss(window,rate1)
    window=evap(window,rate2)
    
    for agent in agent_list:
        
        if change_param :
            
            agent.change_param(param1,param2,param3)
            
        pos=agent.step(window)
        window[int(pos[0,0]),int(pos[0,1])] = color
        
    surf = pg.surfarray.make_surface(window)
    screen.blit(surf, (0, 0))
    pg.display.update()

pg.quit()
        


        
    
