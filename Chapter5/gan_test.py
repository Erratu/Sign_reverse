import torch
from torch import nn
from gan import GAN, Generator
import math
import matplotlib.pyplot as plt
import iisignature

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
#log_interval = 2
sig_level = 3
num_signs = 512

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





