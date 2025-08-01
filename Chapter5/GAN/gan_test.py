import torch
from torch import nn
#from mmd_gan import MMDGAN
from data_gen import create_training_data_gan
from wgan_gp import GAN as WGAN
import iisignature

# Define variables
CUDA = False
batch_size = 64
lr_G = 1e-4
lr_D = 2e-4
num_epochs = 500
loss_function = nn.BCEWithLogitsLoss()
input_dim = 32
nb_ch = 2000
channels = [3,3,3,4,3,7,3]
size_ts = 100

distr_num = 1
channel = channels[distr_num]
sign_dim = iisignature.siglength(channel, 3)+1

#num_classes = 6 # type des données générées
#num_ex = 400
#gen_size = [int((channel*(channel**3 -1))/(channel-1)) for channel in channels]

#mmd_batch_size = 256
#mmd_num_ex = 1000
#mmd_num_epochs = 200


if __name__ == "__main__":
    train_data = create_training_data_gan(size_ts, nb_ch, distr_num)
    data = torch.stack(train_data)
    torch.save(data, "models_saved/wgan_gp/cosine/training_data.pt")
    mean = data.mean(dim=0, keepdim=True) 
    std = data.std(dim=0, keepdim=True)

    torch.save({'mean': mean, 'std': std}, "models_saved/wgan_gp/cosine/stats_gan.pt")
        
    wgan = WGAN(num_epochs, batch_size, sign_dim, input_dim, loss_function, 10, lr_G, lr_D)
    wgan.train_step(data, mean, std, "cosine")
    gen = wgan.generator
    latent_space_samples = torch.randn((batch_size, input_dim))
    gen_data = gen(latent_space_samples) * std + mean
    print(data, gen_data)



