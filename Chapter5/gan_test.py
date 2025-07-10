import torch
from torch import nn
import math
import iisignature
import numpy as np
from gan import GAN

# Define variables
CUDA = False
batch_size = 32
lr_G = 0.001
lr_D = 0.0002
num_epochs = 400
loss_function = nn.BCELoss()
num_classes = 6 # type des données générées
channels = [3,3,3,4,3,8]
gen_size = max([int((channel*(channel**3 -1))/(channel-1)) for channel in channels])
input_dim = 8
sig_level = 3
num_signs = 512
size_ts= 100
sigma = 0.1
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
    f = lambda t: t**4 - t**3 - 5*t**2 - 8*t + 4 + sigma*np.random.normal(size = size_ts)
    g = lambda t: t**4 - t**3 - 5*t**2 - 8*t + 4
    traj = f(times)
    TS = create_TD(False, times, traj)
    return TS

def create_cosine(times):
    # class = 1
    f = lambda t: np.cos(t)+sigma*np.random.normal(size = size_ts)
    g = lambda t: np.cos(t)
    traj = f(times)
    TS = create_TD(False, times, traj)
    return TS

def create_exp(times):
    # class = 2
    f = lambda t: np.exp(-(t-3)**2)+sigma*np.random.normal(size = size_ts)
    g = lambda t: np.exp(-(t-3)**2)
    traj = f(times)
    TS = create_TD(False, times, traj)
    return TS

def create_noisy_circle(times):
    # class = 3
    times = np.linspace(0,2*np.pi,num = size_ts)
    f1 = lambda t: np.cos(t)+sigma*np.random.normal(size = size_ts)
    f2 = lambda t: np.sin(t)+sigma*np.random.normal(size = size_ts)
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

def create_training_data(num_ex_classes):
    times = np.linspace(0,T,num = size_ts)
    data = []
    for class_num, num_ex in enumerate(num_ex_classes):
        for _ in range(num_ex):
            TS = classes[class_num](times)
            signature = torch.from_numpy(iisignature.sig(TS, sig_level)).float()
            if signature.shape[0] != gen_size:
                signature = nn.functional.pad(signature, (0, gen_size - signature.shape[0]))
            data.append((signature, class_num))
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

#gan = GAN(num_epochs, batch_size, num_classes, gen_size, input_dim, loss_function, lr_G, lr_D)
#gan.train_step(train_set)

#G2 = Generator()
#G2.load_state_dict(torch.load("./models_saved/generator_sin.pt"))

def grid_search(train_set):
    # Learning rates
    lrG_values = [1e-4, 2e-4, 5e-4, 1e-3]
    lrD_values = [1e-4, 2e-4, 5e-4, 1e-3]
    # Betas pour Adam
    betas_values = [(0.5, 0.999), (0.4, 0.9), (0.0, 0.9)]
    # Latent space (z) dimension à tester en dernier, d'abord 16 puis 32 puis 64 puis 8
    latent_dim_values = [8, 16, 32, 64]
    # Optionnel : batch size
    batch_size_values = [16, 32, 64, 128]
    size = 16
    for dim in latent_dim_values:
        for beta in betas_values:
            for lrG in lrG_values:
                for lrD in lrD_values:
                    gan = GAN(200, size, num_classes, gen_size, dim, loss_function, lrG, lrD, beta)
                    gan.train_step(train_set)

def grid_search2(train_set):
    # Learning rates
    lrG_values = [5e-4, 1e-3]
    lrD_values = [2e-4, 1e-3]
    latent_dim_values = [8, 16]
    batch_size_values = [32, 64, 128]
    for dim in latent_dim_values:
        for size in batch_size_values:
            for lrG in lrG_values:
                for lrD in lrD_values:
                    gan = GAN(400, size, num_classes, gen_size, dim, loss_function, lrG, lrD)
                    gan.train_step(train_set)

def grid_search_hyperhyper():
    for num_ex in [200,300,400,500,600,700]:
        train_data = create_training_data([num_ex for _ in range(num_classes)])
        for num_e in [100,200,300,400,500,600]:
            gan = GAN(num_e, batch_size, num_classes, gen_size, input_dim, loss_function, lr_G, lr_D)
            gan.train_step(train_data)

if __name__ == "__main__":
    grid_search_hyperhyper()
    #train_data = create_training_data([500 for _ in range(num_classes)])
    #gan = GAN(num_epochs, batch_size, num_classes, gen_size, input_dim, loss_function, lr_G, lr_D)
    #gan.train_step(train_data)





