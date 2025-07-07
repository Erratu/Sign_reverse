import torch
from torch import nn
from gan import GAN, Generator
import math
import matplotlib.pyplot as plt
#import iisignature

# Define variables
CUDA = False
DATA_PATH = './data'
batch_size = 10
num_epochs = 5
num_classes = 6 # type des données générées
channels = 2
gen_size = (channels*(channels**3 -1))/(channels-1)
input_dim = 100
log_interval = 100
lr = 0.001
loss_function = nn.BCELoss()

train_data_length = 200
train_data = []
classes = []

for _ in range(50) :
    data = torch.zeros((train_data_length, 2))
    data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
    data[:, 1] = torch.sin(data[:, 0])
    #sorted_indices = torch.argsort(data[:, 0])
    #data = data[sorted_indices]

    # Calculer la signature d'ordre 3
    #sig_level = 3
    #signature = iisignature.sig(data.numpy(), sig_level)
    #signature.append(data)
    train_data.append(data)
    classes.append(1)

gan = GAN(num_epochs, batch_size, num_classes, (200,2), input_dim, loss_function, loss_function, lr)
gan.train_step(train_data,classes)

latent_space_samples = torch.randn(100, 2)
generated_samples = Generator(latent_space_samples)

generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")






