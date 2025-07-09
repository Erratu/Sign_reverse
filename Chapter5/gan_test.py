import torch
from torch import nn
import math
import iisignature
import numpy as np

# Define variables
CUDA = False
batch_size = 32
lr_G = 0.001
lr_D = 0.0002
num_epochs = 300
loss_function = nn.BCELoss()
num_classes = 6 # type des données générées
channels = 2
gen_size = int((channels*(channels**3 -1))/(channels-1))
input_dim = 8
sig_level = 3
num_signs = 512
size_ts= 100
sigma = 0.1

def create_polynomial(times):
    f = lambda t: t**4 - t**3 - 5*t**2 - 8*t + 4 + sigma*np.random.normal(size = size_ts)
    g = lambda t: t**4 - t**3 - 5*t**2 - 8*t + 4
    idx_ts = np.argsort(times)
    traj = f(times[idx_ts])
    return (times[idx_ts],traj)

def create_cosine(times):
    # class = 1
    f = lambda t: np.cos(t)+sigma*np.random.normal(size = size_ts)
    g = lambda t: np.cos(t)
    idx_ts = np.argsort(times)
    traj = f(times[idx_ts])
    return (times[idx_ts],traj)

def create_exp(times):
    # class = 2
    f = lambda t: np.exp(-(t-3)**2)+sigma*np.random.normal(size = size_ts)
    g = lambda t: np.exp(-(t-3)**2)
    idx_ts = np.argsort(times)
    traj = f(times[idx_ts])
    return (times[idx_ts],traj)

def create_noisy_circle(times):
    # class = 3
    times = np.linspace(0,2*np.pi,num = size_ts)
    f1 = lambda t: np.cos(t)+sigma*np.random.normal(size = size_ts)
    f2 = lambda t: np.sin(t)+sigma*np.random.normal(size = size_ts)
    traj = np.array([f1(times),f2(times)])
    idx_ts = np.argsort(times)
    traj = traj[:,idx_ts]
    times = times[idx_ts]
    return (times,traj)

def create_brown_1D(times):
    # class = 4
    traj = brown(size = size_ts,sig = 1)
    times = np.linspace(0, 1, num = size_ts)
    return (times,traj)

def create_brown_multiD(times):
    # class = 5
    dim = 5
    traj = brown(size = (size_ts,dim),sig = 1).T
    times = np.linspace(0, 1, num = traj.shape[1])
    idx_ts = np.argsort(times)
    times = times[idx_ts]
    return (times,traj)

classes = [create_polynomial,create_cosine,create_exp,create_noisy_circle,create_brown_1D,create_brown_multiD]

def create_training_data(num_ex_classes):
    times = np.linspace(0,T,num = size_ts)
    for class_num, num_ex in enumerate(num_ex_classes):
        data = []
        for i in range(num_ex):
            data.append(classes[class_num](times))

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


if __name__ == "__main__":
    train_data = create_data()
    results = grid_search2(train_data)





