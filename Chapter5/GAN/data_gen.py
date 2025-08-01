import torch
from torch import nn
import math
from signatory import Signature
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt

sig_level = 3
num_signs = 512
T = 5
time_add = True


def data_gen_curve(size_ts, curve=1):
    sigma = 0.1
    times = np.linspace(0,T,num = size_ts)

    if curve == 0:
        f = lambda t: t**4 - t**3 - 5*t**2 - 8*t + 4 + sigma*np.random.normal(size = size_ts)
        idx_ts = np.argsort(times)
        traj = f(times[idx_ts])
    if curve == 1:
        f = lambda t: np.cos(t)+sigma*np.random.normal(size = size_ts)
        idx_ts = np.argsort(times)
        traj = f(times[idx_ts])
    if curve == 2:
        f = lambda t: np.sin(t)+sigma*np.random.normal(size = size_ts)
        idx_ts = np.argsort(times)
        traj = f(times[idx_ts])
    if curve == 3:
        f = lambda t: np.exp(-(t-3)**2)+sigma*np.random.normal(size = size_ts)
        idx_ts = np.argsort(times)
        traj = f(times[idx_ts])
    if curve == 4:
        times = np.linspace(0,2*np.pi,num = size_ts)
        f1 = lambda t: np.cos(t)+sigma*np.random.normal(size = size_ts)
        f2 = lambda t: np.sin(t)+sigma*np.random.normal(size = size_ts)
        traj = np.array([f1(times),f2(times)])
        idx_ts = np.argsort(times)
        traj = traj[:,idx_ts]
        times = times[idx_ts]
    if curve == 5:
        traj = brown(size = size_ts,sig = 1)
        times = np.linspace(0, 1, num = size_ts)
    if curve == 6:
        dim = 5
        traj = brown(size = (size_ts,dim),sig = 1).T
        times = np.linspace(0, 1, num = traj.shape[1])
        idx_ts = np.argsort(times)
        times = times[idx_ts]

    return times, traj

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
    return L, TS

def create_polynomial(size_ts):
    coef_range= (-5,5)
    mult = randint(1,15)
    degree = 5
    noise_std = randint(1,20)
    times = np.linspace(0,T,num = size_ts)

    coefs = np.random.uniform(*coef_range, size=(degree + 1,))
    traj = coefs[0] + sum(coefs[i] * times ** (degree - i) for i in range(1,degree + 1)) + mult*np.random.normal(0, noise_std, size = size_ts)

    return times,traj

def create_cosine(size_ts):
    mult = random()*0.5
    noise_std = random()*0.5
    times = np.random.uniform(0,T, (size_ts,))
    times = np.sort(times)
    # class = 1
    f = lambda t: np.cos(t)+mult*np.random.normal(0, noise_std, size = size_ts)
    traj = f(times)
    
    return times,traj

def create_sine(size_ts):
    mult = random()*0.5
    noise_std = random()*0.5
    times = np.random.uniform(0,T, (size_ts,))
    times = np.sort(times)
    # class = 1
    f = lambda t: np.sin(t)+mult*np.random.normal(0, noise_std, size = size_ts)
    traj = f(times)
    return times,traj

def create_exp(size_ts):
    # class = 2
    mult = random()*0.5
    noise_std = random()*0.5
    mu_range=(4, 6)
    sigma_range=(0.1, 1.0)
    mu = np.random.uniform(*mu_range)       
    sigma = np.random.uniform(*sigma_range)
    times = np.random.uniform(0,T, (size_ts,))
    times = np.sort(times)

    f = lambda t: np.exp(-((t - mu) ** 2) / (2 * sigma ** 2)) + mult*np.random.normal(0, noise_std, size = size_ts)
    traj = f(times)
    return times,traj

def create_noisy_circle(size_ts):
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

def create_brown_1D(size_ts):
    # class = 4
    traj = brown(size = size_ts,sig = 1)
    times = np.linspace(0, 1, num = size_ts)
    return times,traj

def create_brown_multiD(size_ts):
    # class = 5
    dim = 5
    traj = brown(size = (size_ts,dim),sig = 1).T
    times = np.linspace(0, 1, num = traj.shape[1])
    return times,traj

classes = [create_polynomial,create_cosine,create_sine,create_exp,create_noisy_circle,create_brown_1D,create_brown_multiD]

def create_training_data_cgan(size_ts, num_ex_classes, gen_size):
    data = []
    for class_num, num_ex in enumerate(num_ex_classes):
        for _ in range(num_ex):
            times,TS = classes[class_num](size_ts)
            path = np.column_stack((times, TS))
            signature = torch.from_numpy(iisignature.sig(path, sig_level)).float()
            if signature.shape[0] != max(gen_size):
                signature = nn.functional.pad(signature, (0, max(gen_size) - signature.shape[0]))
            data.append((signature, class_num))
    return data

def create_training_data_gan(size_ts, nb_ch, distr_num):
    data = []
    signature_TS = Signature(depth = 3,scalar_term= True)
    for _ in range(nb_ch):
        times,traj = data_gen_curve(size_ts, curve=distr_num)
        L, TS = create_TD(False, times, traj, time_add = True)
        signature = signature_TS(torch.from_numpy(TS)[None]).float().squeeze(0)
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

def TS_graph(distr_num, size_ts):
    for _ in range(2):
        times,TS = data_gen_curve(size_ts, curve=distr_num)
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
    TS_graph(1, 600)

    #for num in range(6):
    #   TS = classes[num](times)
    #   signature = torch.from_numpy(iisignature.sig(TS, sig_level)).float()
        