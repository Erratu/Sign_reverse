import torch
from torch import nn
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wgan_gp_loss_inv import GAN as WGAN
from data_gen import create_training_data_gan

# Define variables
CUDA = False
batch_size = 64
lr_G = 1e-5
lr_D = 1e-5
num_epochs = 600
loss_function = nn.BCEWithLogitsLoss()
input_dim = 32
nb_ch = 2000
channels = [3,3,3,3,4,3,7]
size_ts = 100

distr_num = 5
channel = channels[distr_num]


if __name__ == "__main__":
    train_data = create_training_data_gan(size_ts, nb_ch, distr_num)
    data = torch.stack(train_data)
    torch.save(data, "models_saved/wgan_gp/brown1D/training_data_inv.pt")
    mean = data.mean(dim=0, keepdim=True) 
    std = data.std(dim=0, keepdim=True)

    torch.save({'mean': mean, 'std': std}, "models_saved/wgan_gp/brown1D/stats_gan_inv.pt")
       
    wgan = WGAN(num_epochs, batch_size, channel, input_dim, loss_function, 10, lr_G, lr_D)
    wgan.train_step(data, mean, std, "brown1D")

    latent_space_samples = torch.randn((batch_size, input_dim))
    gen_inv_stand = wgan.generator(latent_space_samples) 
    sign_inv = torch.where(std != 0, gen_inv_stand * std + mean, gen_inv_stand + mean)

    X_real_stand = torch.where(std != 0, (data[:64] - mean) / std, data[:64] - mean)
    print("mean real/gen:", data[:64].mean(axis=0), sign_inv.mean(axis=0))
    print("std  real/gen:", data[:64].std(axis=0), sign_inv.std(axis=0))

    print("MSE between reps:", np.mean((X_real_stand-gen_inv_stand).detach().numpy()**2))
    cos_sim = torch.nn.functional.cosine_similarity(X_real_stand, gen_inv_stand, dim=1)
    mean_cos = cos_sim.mean()
    print("cosine similarity :",mean_cos)

    # Covariance matrices car standardisé => covariance = corrélation
    cov_real = (X_real_stand.T @ X_real_stand) / (X_real_stand.shape[0]-1)
    cov_gen  = (gen_inv_stand.T @ gen_inv_stand) / (gen_inv_stand.shape[0]-1)

    mse_corr = ((cov_real - cov_gen)**2).mean()
    print("MSE corrélations:", mse_corr.item())



