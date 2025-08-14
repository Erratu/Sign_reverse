import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from signatory import Signature
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_gen import data_gen_curve, create_TD
from wgan_gp_loss_inv import Generator as Gen_inv
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

    device = "cpu"
    # test avec l'inv des deux gan et comparaison
    signature_TS = Signature(depth = 3,scalar_term= True).to(device)

    times,traj = data_gen_curve(size_ts, curve=1)
    L, TS = create_TD(False, times, traj, time_add = True)
    signature = signature_TS(torch.from_numpy(TS)[None].to(device))

    sign_dim = signature.shape[1]

    data_stats = torch.load("models_saved/wgan_gp/cosine/stats_gan.pt")
    mean = data_stats['mean']
    std = data_stats['std']

    gen_inv = Gen_inv(input_dim, sign_dim)
    gen_inv.load_state_dict(torch.load('models_saved/wgan_gp/cosine/inv_G_model_1.pt'), strict=False) 

    latent_space_samples = torch.randn((batch_size, input_dim))
    sign = gen_inv(latent_space_samples) 
    sign = torch.where(std != 0, sign * std + mean, sign + mean)

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

    SA = SeigalAlgo(size_ts, len_base, chan, real_chan, depth, n_recons, size_base, time_chan=True, sig_TS=sign[0].detach().unsqueeze(0))
    base = SA.define_base(base_name).flip([-2,-1])
    # Retrieve A from depth 3 signature. "par" is deprecated for the moment. If cuda is available, compute automatically from cuda.
    A = SA.retrieve_coeff_base(base, par = 1, limits = limits, lrs = lrs, opt = optim, params = params)
    # Recompose signal from A
    recomposed_signal = np.matmul(A.detach().numpy(),base[0].detach().numpy().T)

    x_axis = np.linspace(0,10,num = recomposed_signal.shape[1])
    plt.plot(x_axis,traj-traj[0])
    plt.plot(x_axis,np.flip(recomposed_signal[2,:]))
    plt.legend(['truth_signal','GAN_recon_signal','reconstruction_signal'])
    plt.savefig("Inv_results/reconstruction_cos_pw1.png")
    plt.show()
    print(np.mean(traj-traj[0]-recomposed_signal[2,:]))
    plt.plot(x_axis,L)
    plt.plot(x_axis,np.flip(recomposed_signal[1,:]))
    plt.legend(['truth_length','GAN_length','reconstruction_length'])
    plt.show()
