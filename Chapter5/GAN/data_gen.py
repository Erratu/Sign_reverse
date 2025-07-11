import torch
from torch import nn
import math
import iisignature
import numpy as np
from random import random, randint

sig_level = 3
num_signs = 512
size_ts= 100
T = 5
time_add = True


def brown(size,sig=1):
    jump = np.random.normal(loc = 0, scale = sig, size = size)
    if type(size) is not int:
        jump = jump.reshape(size)
    return jump.cumsum(axis=0)

def create_TD(multichan, times, traj):
    if not multichan:
        L = np.abs(traj[1:]-traj[:-1]).cumsum()
        L = np.insert(L,0,0)
        if time_add:
            TS = np.array([times,L,traj-traj[0]]).T
        else:
            TS = np.array([L,traj-traj[0]]).T
        #print("1D:",TS.shape)
    else:
        L = np.linalg.norm(traj[:,1:]-traj[:,:-1],axis=0).cumsum(axis=0)
        L = np.insert(L,0,0)
        if time_add:
            TS = torch.cat((torch.tensor(times)[None],torch.tensor(L)[None],torch.tensor(traj-traj[:,0,None])),axis=0).numpy().T
        else:
            TS = torch.cat((torch.tensor(L)[None],torch.tensor(traj-traj[:,0,None])),axis=0).numpy().T
        #print("MD:",TS.shape)
    return TS

def create_polynomial(times):
    coef_range= (-5,5)
    mult = random()
    degree = 4
    noise_std = random()

    coefs = np.random.uniform(*coef_range, size=(degree + 1,))
    traj = sum(coefs[i] * times ** (degree - i) for i in range(degree + 1)) + mult*np.random.normal(0, noise_std, size = size_ts)
    TS = create_TD(False, times, traj)
    return TS

def create_cosine(times):
    mult = random()
    coeff = randint(1,5)
    noise_std = random()

    # class = 1
    f = lambda t: coeff*np.cos(t)+mult*np.random.normal(0, noise_std, size = size_ts)
    traj = f(times)
    TS = create_TD(False, times, traj)
    return TS

def create_exp(times):
    # class = 2
    mult = random()
    noise_std = random()
    amplitude_range=(0.5, 2.0)
    mu_range=(-1, 1)
    sigma_range=(0.1, 1.0)
    a = np.random.uniform(*amplitude_range) 
    mu = np.random.uniform(*mu_range)       
    sigma = np.random.uniform(*sigma_range)

    f = lambda t: a * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) + mult*np.random.normal(0, noise_std, size = size_ts)
    traj = f(times)
    TS = create_TD(False, times, traj)
    return TS

def create_noisy_circle(times):
    # class = 3
    mult = random()
    noise_std = random()
    radius_range=(0.8, 1.2)
    r = np.random.uniform(*radius_range)

    times = np.linspace(0,2*np.pi,num = size_ts)
    f1 = lambda t: r*np.cos(t)+mult*np.random.normal(0, noise_std, size = size_ts)
    f2 = lambda t: r*np.sin(t)+mult*np.random.normal(0, noise_std, size = size_ts)
    traj = np.array([f1(times),f2(times)])
    idx_ts = np.argsort(times)
    traj = traj[:,idx_ts]
    times = times
    TS = create_TD(True, times, traj)
    return TS

def create_brown_1D(times):
    # class = 4
    traj = brown(size = size_ts,sig = 1)
    times = np.linspace(0, 1, num = size_ts)
    TS = create_TD(False, times, traj)
    return TS

def create_brown_multiD(times):
    # class = 5
    dim = 5
    traj = brown(size = (size_ts,dim),sig = 1).T
    times = np.linspace(0, 1, num = traj.shape[1])
    times = times
    TS = create_TD(True, times, traj)
    return TS

classes = [create_polynomial,create_cosine,create_exp,create_noisy_circle,create_brown_1D,create_brown_multiD]

def create_training_data_cgan(num_ex_classes, gen_size):
    times = np.linspace(0,T,num = size_ts)
    data = []
    for class_num, num_ex in enumerate(num_ex_classes):
        for _ in range(num_ex):
            TS = classes[class_num](times)
            signature = torch.from_numpy(iisignature.sig(TS, sig_level)).float()
            if signature.shape[0] != max(gen_size):
                signature = nn.functional.pad(signature, (0, max(gen_size) - signature.shape[0]))
            data.append((signature, class_num))
    return data

def create_training_data_gan(nb_ch, distr_num):
    times = np.linspace(0,T,num = size_ts)
    data = []
    for _ in range(nb_ch):
        TS = classes[distr_num](times)
        signature = torch.from_numpy(iisignature.sig(TS, sig_level)).float()
        data.append(signature)
    return data

def create_data():
    torch.manual_seed(111)
    data = []
    for _ in range(num_signs):
        train_data_length = 1024
        train_data = torch.zeros((train_data_length, 2))
        train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
        train_data[:, 1] = torch.sin(train_data[:, 0])
        sorted_indices = torch.argsort(train_data[:, 0])
        train_data = train_data[sorted_indices]
        # Calculer la signature d'ordre 3
        signature = torch.from_numpy(iisignature.sig(train_data.numpy(), sig_level)).float()
        data.append(signature)
    
    train_labels = torch.zeros(train_data_length)
    train_set = [
        (data[i], train_labels[i]) for i in range(num_signs)
    ]
    return train_set