import torch
from torch import nn
from data_gen import create_training_data_gan
from unused.wgan_gp_loss_sup import GAN as WGAN

# Define variables
CUDA = False
batch_size = 64
lr_G = 1e-5
lr_D = 1e-5
num_epochs = 600
loss_function = nn.BCEWithLogitsLoss()
input_dim = 32
nb_ch = 2000
channels = [3,3,3,4,3,7,3]
size_ts = 100

distr_num = 1
channel = channels[distr_num]


if __name__ == "__main__":
    train_data = create_training_data_gan(size_ts, nb_ch, distr_num)
    data = torch.stack(train_data)
    torch.save(data, "models_saved/wgan_gp/cosine/training_data_sup.pt")
    mean = data.mean(dim=0, keepdim=True) 
    std = data.std(dim=0, keepdim=True)

    torch.save({'mean': mean, 'std': std}, "models_saved/wgan_gp/cosine/stats_gan_sup.pt")
       
    wgan = WGAN(num_epochs, batch_size, channel, input_dim, loss_function, 10, lr_G, lr_D)
    wgan.train_step(data, mean, std, "cosine")



