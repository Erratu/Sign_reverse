import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from data_gen import create_training_data_gan
from wgan_gp_loss_inv import Generator as Gen_inv
from unused.wgan_gp_loss_sup import Generator as Gen_sup
from unused.wgan_gp import Generator as Gen
from signatory import signature_channels

# Define variables
CUDA = False
batch_size = 64
lr_G = 1e-5
lr_D = 1e-5
num_epochs = 800
loss_function = nn.BCEWithLogitsLoss()
input_dim = 32
nb_ch = 2000
channels = [3,3,3,4,3,7,3]
size_ts = 100

distr_num = 1
channel = channels[distr_num]

#num_classes = 6 # type des données générées
#num_ex = 400
#gen_size = [int((channel*(channel**3 -1))/(channel-1)) for channel in channels]

#mmd_batch_size = 256
#mmd_num_ex = 1000
#mmd_num_epochs = 200


if __name__ == "__main__":
    train_data = create_training_data_gan(size_ts, nb_ch, distr_num)
    data = torch.stack(train_data)
    mean = data.mean(dim=0, keepdim=True) 
    std = data.std(dim=0, keepdim=True)

    sign_dim = signature_channels(channel, 3, scalar_term=True)
    
    num_epochs = 800
    gen_inv = Gen_inv(input_dim, sign_dim)
    gen_inv.load_state_dict(torch.load('models_saved/wgan_gp/cosine/inv_G_model.pt'), strict=False)
    latent_space_samples = torch.randn((batch_size, input_dim))
    gen_inv_stand = gen_inv(latent_space_samples) 
    sign_inv = torch.where(std != 0, gen_inv_stand * std + mean, gen_inv_stand + mean)

    num_epochs = 500
    gen_inv = Gen_inv(input_dim, sign_dim)
    gen_inv.load_state_dict(torch.load('models_saved/wgan_gp/cosine/inv_G_model.pt'), strict=False)
    latent_space_samples = torch.randn((batch_size, input_dim))
    gen_inv_stand = gen_inv(latent_space_samples) 
    sign_inv = torch.where(std != 0, gen_inv_stand * std + mean, gen_inv_stand + mean)

    num_epochs = 1000
    gen_inv = Gen_inv(input_dim, sign_dim)
    gen_inv.load_state_dict(torch.load('models_saved/wgan_gp/cosine/inv_G_model.pt'), strict=False)
    latent_space_samples = torch.randn((batch_size, input_dim))
    gen_inv_stand = gen_inv(latent_space_samples) 
    sign_inv = torch.where(std != 0, gen_inv_stand * std + mean, gen_inv_stand + mean)

    print(data, sign, sign_inv)
    X_real_stand = torch.where(std != 0, (data[:64] - mean) / std, data[:64] - mean)
    print("mean real/gen:", data[:64].mean(axis=0), sign_inv.mean(axis=0))
    print("std  real/gen:", data[:64].std(axis=0), sign_inv.std(axis=0))

    print("sup")
    print("MSE between reps:", np.mean((X_real_stand-gen_sup_stand).detach().numpy()**2))
    cos_sim = torch.nn.functional.cosine_similarity(X_real_stand, gen_sup_stand, dim=1)
    mean_cos = cos_sim.mean()
    print("cosine similarity :",mean_cos)

    # Covariance matrices car standardisé => covariance = corrélation
    cov_real = (X_real_stand.T @ X_real_stand) / (X_real_stand.shape[0]-1)
    cov_gen  = (gen_sup_stand.T @ gen_sup_stand) / (sign_sup.shape[0]-1)

    mse_corr = ((cov_real - cov_gen)**2).mean()
    print("MSE corrélations:", mse_corr.item())

    print("inv")
    print("MSE between reps:", np.mean((X_real_stand-gen_inv_stand).detach().numpy()**2))
    cos_sim = torch.nn.functional.cosine_similarity(X_real_stand, gen_inv_stand, dim=1)
    mean_cos = cos_sim.mean()
    print("cosine similarity :",mean_cos)

    # Covariance matrices car standardisé => covariance = corrélation
    cov_real = (X_real_stand.T @ X_real_stand) / (X_real_stand.shape[0]-1)
    cov_gen  = (gen_inv_stand.T @ gen_inv_stand) / (sign_inv.shape[0]-1)

    mse_corr = ((cov_real - cov_gen)**2).mean()
    print("MSE corrélations:", mse_corr.item())

    print("ori")
    print("MSE between reps:", np.mean((X_real_stand-gen_data_stand).detach().numpy()**2))
    cos_sim = torch.nn.functional.cosine_similarity(X_real_stand, gen_data_stand, dim=1)
    mean_cos = cos_sim.mean()
    print("cosine similarity :",mean_cos)

    # Covariance matrices car standardisé => covariance = corrélation
    cov_real = (X_real_stand.T @ X_real_stand) / (X_real_stand.shape[0]-1)
    cov_gen  = (gen_data_stand.T @ gen_data_stand) / (gen_data_stand.shape[0]-1)

    mse_corr = ((cov_real - cov_gen)**2).mean()
    print("MSE corrélations:", mse_corr.item())