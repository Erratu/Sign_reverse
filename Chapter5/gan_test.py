import torch
from torch import nn
from gan import GAN, Generator
import math
import matplotlib.pyplot as plt
#import iisignature

# Define variables
CUDA = False
batch_size = 32
lr = 0.001
num_epochs = 300
loss_function = nn.BCELoss()
num_classes = 6 # type des données générées
channels = 2
gen_size = (channels*(channels**3 -1))/(channels-1)
input_dim = 100
log_interval = 2

torch.manual_seed(111)

train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]

gan = GAN(num_epochs, batch_size, num_classes, (200,2), input_dim, loss_function, lr, log_interval)
gan.train_step(train_set)

G2 = Generator()
G2.load_state_dict(torch.load("./models_saved/generator_sin.pt"))

latent_space_samples = torch.randn(100, 2)
generated_samples = G2(latent_space_samples)
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
plt.show()









