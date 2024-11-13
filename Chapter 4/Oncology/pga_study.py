# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:23:41 2024

@author: popym
"""

import signatory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

data = {}
for b in [0.02,0.07,0.14,0.21,0.28]:
  df = pd.read_csv("H:/Mon Drive/Etude_Cancer/Simul/benign_b"+str(b)+".csv",sep = ";")
  df_numpy = df.to_numpy().T[:,:,None]
  data[str(b)]= torch.tensor(pd.read_csv("H:/Mon Drive/Etude_Cancer/Simul/benign_b"+str(b)+".csv",sep = ";").to_numpy().T[:,:,None])
  
time = torch.tensor([300*i for i in range(7)])[None,:,None]
times = time.repeat(1000,1,1)

for arrays in data:
  data[arrays]=torch.cat((data[arrays][1:],times),dim = -1)
  
data_ben = torch.cat((data['0.02'],data['0.07'],data['0.14'],data['0.21'],data['0.28']))
data_ben[:,:,1] = data_ben[:,:,1]
data_ben[:,:,0] = (data_ben[:,:,0].T-data_ben[:,:,0].mean(axis=1)).T

import os
import re


def starts_with_benign(s):
    return s.startswith("benign")
classif = []
data_ben_mal = {}
directory = 'H:/Mon Drive/Etude_Cancer/Simul'
for filename in os.listdir(directory):
    if starts_with_benign(filename):
      i = 1
    else:
      f = os.path.join(directory, filename)
      data_ben_mal[filename]= torch.tensor(pd.read_csv(f,sep = ";").to_numpy().T[:,:,None])


time = torch.tensor([300*i for i in range(7)])[None,:,None]

for arrays in data_ben_mal:
  times = time.repeat(data_ben_mal[arrays].shape[0],1,1)
  data_ben_mal[arrays]=torch.cat((data_ben_mal[arrays],times),dim = -1)
  
data_mal = torch.cat(tuple([data_ben_mal[filename] for filename in data_ben_mal.keys()]))
data_mal[:,:,1] = data_mal[:,:,1]/1800
data_mal[:,:,0] = (data_mal[:,:,0].T-data_mal[:,:,0].mean(axis=1)).T

from fct_pga import compute_pga, compute_projection, visualisation_all, visualisation_normales



##### Rescaling #####
rescale = True

M = data_ben.max()
m = data_ben.min()

f = lambda X: (X-m)/(M-m)

if rescale:
  data_ben = f(data_ben)
  data_mal = f(data_mal)
  
  
t_principal_directions, signatures_ben = compute_pga(path = data_ben,sig_level=2,tangent=False,n_components = 5)

np.save('principal_directions_all_sig3_n10.npy',t_principal_directions)


idx_ben = [i for i in range(200)]
for j in range(1,5):
  idx_ben  += [1000*j + i for i in range(200)]

projections_ben = compute_projection(t_principal_directions,data_ben[:100],channels = data_ben.shape[-1],depth = 3)
visualisation_normales(projections_ben,colour = ['blue']*100)#len(idx_ben))

projections_mal = compute_projection(t_principal_directions,data_mal[:100],channels = data_mal.shape[-1],depth = 3)
visualisation_all(projections_norm=projections_ben,projections_anorm=projections_mal)


