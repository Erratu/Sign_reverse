import torch
from torch import nn
import math
import iisignature
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt

sig_level = 3
num_signs = 512
size_ts= 600
T = 5
time_add = True


def brown(size,sig=1):
    jump = np.random.normal(loc = 0, scale = sig, size = size)
    if type(size) is not int:
        jump = jump.reshape(size)
    return jump.cumsum(axis=0)

def create_TD(multichan, times, traj, time_add = True):
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

def create_polynomial():
    coef_range= (-5,5)
    mult = randint(1,15)
    degree = 5
    noise_std = randint(1,20)
    times = np.random.uniform(0,10, (size_ts,))
    times = np.sort(times)

    coefs = np.random.uniform(*coef_range, size=(degree + 1,))
    traj = coefs[0] + sum(coefs[i] * times ** (degree - i) for i in range(1,degree + 1)) + mult*np.random.normal(0, noise_std, size = size_ts)

    return times,traj

def create_cosine():
    mult = random()*0.5
    coeff = randint(1,5)
    noise_std = random()*0.5
    times = np.random.uniform(0,10, (size_ts,))
    times = np.sort(times)
    # class = 1
    f = lambda t: coeff*np.cos(random()*t)+mult*np.random.normal(0, noise_std, size = size_ts)
    traj = f(times)
    
    return times,traj

def create_exp():
    # class = 2
    mult = random()*0.5
    noise_std = random()*0.5
    amplitude_range=(1, 6)
    mu_range=(4, 6)
    sigma_range=(0.1, 1.0)
    a = np.random.uniform(*amplitude_range) 
    mu = np.random.uniform(*mu_range)       
    sigma = np.random.uniform(*sigma_range)
    times = np.random.uniform(0,10, (size_ts,))
    times = np.sort(times)

    f = lambda t: a * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) + mult*np.random.normal(0, noise_std, size = size_ts)
    traj = f(times)
    return times,traj

def create_noisy_circle():
    # class = 3
    radius_range=(0.1, 1)
    r = np.random.uniform(*radius_range)
    mult = random()*0.5*r
    noise_std = random()*0.5*r

    times = np.random.uniform(0,2*np.pi, (size_ts,))
    times = np.sort(times)
    f1 = lambda t: r*np.cos(t)+mult*np.random.normal(0, noise_std, size = size_ts)
    f2 = lambda t: r*np.sin(t)+mult*np.random.normal(0, noise_std, size = size_ts)
    traj = np.array([f1(times),f2(times)])
    return times,traj

def create_brown_1D():
    # class = 4
    traj = brown(size = size_ts,sig = 1)
    times = np.random.uniform(0,10, (size_ts,))
    times = np.sort(times)
    return times,traj

def create_brown_multiD():
    # class = 5
    dim = 5
    traj = brown(size = (size_ts,dim),sig = 1).T
    times = np.random.uniform(0,10, (traj.shape[1],))
    times = np.sort(times)
    return times,traj

classes = [create_polynomial,create_cosine,create_exp,create_noisy_circle,create_brown_1D,create_brown_multiD]

def create_training_data_cgan(num_ex_classes, gen_size):
    data = []
    for class_num, num_ex in enumerate(num_ex_classes):
        for _ in range(num_ex):
            times,TS = classes[class_num]()
            path = np.column_stack((times, TS))
            signature = torch.from_numpy(iisignature.sig(path, sig_level)).float()
            if signature.shape[0] != max(gen_size):
                signature = nn.functional.pad(signature, (0, max(gen_size) - signature.shape[0]))
            data.append((signature, class_num))
    return data

def create_training_data_gan(nb_ch, distr_num):
    data = []
    for _ in range(nb_ch):
        times,TS = classes[distr_num]()
        path = np.column_stack((times, TS))
        signature = torch.from_numpy(iisignature.sig(path, sig_level)).float()
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

def TS_graph(distr_num):
    for _ in range(2):
        times,TS = classes[distr_num]()
        dim = len(TS.shape)
        if dim != 1 : 
            dim = TS.shape[0]
            for D in range(dim):
                plt.plot(times,TS[D])
        else:
            plt.plot(times, TS)
        
    plt.legend([f'TS_{i}' for i in range(4*dim)])
    plt.show()


if __name__ == "__main__":
    TS_graph(0)

    #for num in range(6):
    #   TS = classes[num](times)
    #   signature = torch.from_numpy(iisignature.sig(TS, sig_level)).float()
        