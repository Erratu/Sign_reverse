# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:54:52 2024

@author: w
"""

import numpy as np
import torch
import pandas as pd
import birdepy as bd
from scipy.stats import poisson


'''
Fonction de générations de process
'''

def generate_cox_process(b,d,transition = False):
    tt = 0
    jump_times = np.array([300*i for i in range(7)])
    transition_time = jump_times[-1]
    
    if transition:
        b_ben = d
        d_ben = d
        transition_time = np.random.uniform(400,1500)
        jump_times_b = jump_times[jump_times<transition_time]
        np.insert(jump_times_b,jump_times_b.shape[0],transition_time)
        jump_times_m = jump_times[jump_times>transition_time]-transition_time
        np.insert(jump_times_m,0, 0)
    
    ### Cells number génération using birdepy
    if transition:
        cell_pop_b = bd.simulate.discrete([b_ben,d_ben], 'linear', z0 = 1e9, times = jump_times_b, method = 'gwa')
        cell_pop_m = bd.simulate.discrete([b,d], 'linear', z0 = cell_pop_b[-1], times = jump_times_m, method = 'gwa')
        cell_pop = cell_pop_b+cell_pop_m
    else:
        
        biom_traj = bd.simulate.discrete([b,d,alpha], 'linear-migration', z0 = poisson.rvs(mu=100, loc=0, size=1, random_state=None), times = jump_times, method = 'gwa')
    
    traj_cell = torch.tensor(np.array([jump_times,cell_pop]).T)[None]
    if b=!0:
        int_cox = (traj_cell[:,:,1]*d*1.4*1e-4)/(33+d-b)
        biom_traj = np.random.poisson(lam = int_cox)
    traj_biom = torch.tensor(np.array([jump_times,biom_traj[0]]).T)[None]
    return traj_biom, transition_time, traj_cell