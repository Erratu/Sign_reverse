import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from data_gen import create_training_data_gan, data_gen_curve, create_TD
from wgan_gp_loss_inv import GAN as WGAN_inv
from wgan_gp import GAN as WGAN
from Algo_Seigal_inverse_path2 import SeigalAlgo

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
    torch.save(data, "models_saved/wgan_gp/cosine/training_data.pt")
    mean = data.mean(dim=0, keepdim=True) 
    std = data.std(dim=0, keepdim=True)

    #torch.save({'mean': mean, 'std': std}, "models_saved/wgan_gp/cosine/stats_gan.pt")
        
    wgan = WGAN(num_epochs, batch_size, channel, input_dim, loss_function, 10, lr_G, lr_D)
    #wgan.train_step(data, mean, std, "cosine")
    #gen = wgan.generator

    latent_space_samples = torch.randn((batch_size, input_dim))
    gen_data = gen(latent_space_samples) * std + mean

    #wgan_inv = WGAN_inv(num_epochs, batch_size, channel, input_dim, loss_function, 10, lr_G, lr_D)
    #wgan_inv.train_step(data, mean, std, "cosine")
    #gen_inv = wgan_inv.generator
    
    latent_space_samples = torch.randn((batch_size, input_dim))
    gen_inv_data = gen_inv(latent_space_samples) * std + mean

    print(data, gen_data, gen_inv_data)

    X_real_flat = data.reshape(-1, data.shape[-1])  # shape ((E*T), D)
    gen_data_flat  = gen_data.reshape(-1, data.shape[-1])
    gen_inv_flat  = gen_inv_data.reshape(-1, data.shape[-1])
    print("mean real/gen:", X_real_flat.mean(axis=0), gen_data_flat.mean(axis=0), gen_inv_flat.mean(axis=0))
    print("std  real/gen:", X_real_flat.std(axis=0), gen_data_flat.std(axis=0), gen_inv_flat.std(axis=0))
    print("MSE between reps:", np.mean((X_real_flat-gen_data_flat)**2))

    # test avec l'inv des deux gan et comparaison
    times,traj = data_gen_curve(size_ts, curve=1)
    L, TS = create_TD(False, times, traj, time_add = True)

    latent_space_samples = torch.randn((batch_size, input_dim))
    sign = gen(latent_space_samples) * std + mean
    latent_space_samples = torch.randn((batch_size, input_dim))
    sign_inv = gen_inv(latent_space_samples) * std + mean

    len_base = size_ts-1
    depth = 3
    n_recons = 2
    real_chan = 0
    size_base = len_base+1

    #Number of iteration for optimisation
    limits = 30000
    #Learning rate (there is a patience schedule)
    lrs = 1e-2

    #Available optimizers : "Adam", "AdamW" and "LBFGS"
    optim = "AdamW"
    # params are [lambda_length, lambda_frontier, lambda_levy, lambda_ridge]
    # [1,5,0,1] , [5,1,0,1] or the same with lambda_ridge = 0 worked well
    params = [1,5,0,0]
    chan = TS.shape[1]

    #Available base: "PwLinear", "BSpline", "Fourier"
    base_name = "PwLinear"

    A_init = torch.load('Inv_results/original_A_cos_3.pt')

    SA = SeigalAlgo(size_ts, len_base, chan, real_chan, depth, n_recons, size_base, time_chan=True, sig_TS=sign[0].detach().unsqueeze(0), A_init=A_init)
    base = SA.define_base(base_name).flip([-2,-1])
    # Retrieve A from depth 3 signature. "par" is deprecated for the moment. If cuda is available, compute automatically from cuda.
    A = SA.retrieve_coeff_base(base, par = 1, limits = limits, lrs = lrs, opt = optim, params = params)
    # Recompose signal from A
    recomposed_signal = np.matmul(A.detach().numpy(),base[0].detach().numpy().T)

    SA = SeigalAlgo(size_ts, len_base, chan, real_chan, depth, n_recons, size_base, time_chan=True, sig_TS=sign_inv[0].detach().unsqueeze(0), A_init=A_init)
    base = SA.define_base(base_name).flip([-2,-1])
    # Retrieve A from depth 3 signature. "par" is deprecated for the moment. If cuda is available, compute automatically from cuda.
    A = SA.retrieve_coeff_base(base, par = 1, limits = limits, lrs = lrs, opt = optim, params = params)
    # Recompose signal from A
    recomposed_signal_inv = np.matmul(A.detach().numpy(),base[0].detach().numpy().T)

    x_axis = np.linspace(0,10,num = recomposed_signal.shape[1])
    plt.plot(x_axis,traj-traj[0])
    plt.plot(x_axis,np.flip(recomposed_signal[2,:]))
    plt.plot(x_axis,np.flip(recomposed_signal_inv[2,:]))
    plt.legend(['truth_signal','GAN_recon_signal','reconstruction_signal'])
    plt.savefig("Inv_results/reconstruction_cos_pw1.png")
    plt.show()
    print(np.mean(traj-traj[0]-recomposed_signal[2,:]))
    plt.plot(x_axis,L)
    plt.plot(x_axis,np.flip(recomposed_signal[1,:]))
    plt.plot(x_axis,np.flip(recomposed_signal_inv[1,:]))
    plt.legend(['truth_length','GAN_length','reconstruction_length'])
    plt.show()
