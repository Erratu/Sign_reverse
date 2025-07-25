import numpy as np
from Algo_Seigal_inverse_path2 import SeigalAlgo
import matplotlib.pyplot as plt
import torch
from signatory import Signature

from wgan_gp import Generator
from data_gen import create_cosine, create_TD

batch_size = 32
input_dim = 32
nb_ch = 2000
device = "cpu"
size_ts= 600

def test_inverse_GAN(nb_ch, size_ts, multichan, time_add=True):
    """montre le graphe de la TS associée à la signature créée par le GAN.

    Args:
        nb_ch (int): the number of data to use to train the Generator
        size_ts (int): the size of the TS to recompose
        multichan (bool): If the test data are multidimensional or not
        time_add (bool): If the data are time-augmented. Defaults to true.
    """

    signature_TS = Signature(depth = 3,scalar_term= True).to(device)
    #data = []
    #for _ in range(nb_ch):
    #    times,traj = create_polynomial(size_ts)
    #    L, TS = create_TD(multichan, times, traj, time_add = True)
    #    signature = signature_TS(torch.from_numpy(TS)[None].to(device))
    #    data.append(signature)
    
    times,traj = create_cosine(size_ts)
    L, TS = create_TD(multichan, times, traj, time_add = True)
    signature = signature_TS(torch.from_numpy(TS)[None].to(device))
    sign_dim = signature.shape[1]


    data_stats = torch.load("models_saved/wgan_gp/cosine/stats_gan.pt")
    mean = data_stats['mean']
    std = data_stats['std']

    generator = Generator(input_dim, sign_dim)
    generator.load_state_dict(torch.load('models_saved/wgan_gp/cosine/G_model.pt'), strict=False)
    latent_space_samples = torch.randn((batch_size, input_dim))
    sign = generator(latent_space_samples) * std + mean

    print(signature, sign)

    len_base = size_ts-1
    depth = 3
    n_recons = 2
    real_chan = 0
    size_base = len_base+1

    #Number of iteration for optimisation
    limits = 20000
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

    
    SA = SeigalAlgo(size_ts, len_base, chan, real_chan, depth, n_recons, size_base, time_chan=True, sig_TS=sign[0].unsqueeze(0))
    base = SA.define_base(base_name).flip([-2,-1])
    # Retrieve A from depth 3 signature. "par" is deprecated for the moment. If cuda is available, compute automatically from cuda.
    A = SA.retrieve_coeff_base(base, par = 1, limits = limits, lrs = lrs, opt = optim, params = params)
    # Recompose signal from A
    recomposed_signal = np.matmul(A.detach().numpy(),base[0].detach().numpy().T)

    SA = SeigalAlgo(size_ts, len_base, chan, real_chan, depth, n_recons, size_base, time_chan=True, sig_TS=signature)
    base = SA.define_base(base_name).flip([-2,-1])
    # Retrieve A from depth 3 signature. "par" is deprecated for the moment. If cuda is available, compute automatically from cuda.
    A_original = SA.retrieve_coeff_base(base, par = 1, limits = limits, lrs = lrs, opt = optim, params = params)
    # Recompose signal from A
    recomposed_signal_original = np.matmul(A_original.detach().numpy(),base[0].detach().numpy().T)
    #See the result

    if not multichan:
        if time_add:
            i = 1
        else:
            i=0
        x_axis = np.linspace(0,1,num = recomposed_signal.shape[1])
        plt.plot(x_axis,recomposed_signal_original)
        plt.plot(x_axis,np.flip(recomposed_signal[i+1,:]))
        plt.legend(['truth_signal','reconstruction_signal'])
        plt.savefig("Inv_results/reconstruction_gan_cos_comp.png")
        plt.show()
        print(np.mean(traj-traj[0]-recomposed_signal[i+1,:]))

        plt.plot(x_axis,L)
        plt.plot(x_axis,np.flip(recomposed_signal[1,:]))
        plt.legend(['truth_length','reconstruction_length'])
        plt.show()

    else:
        x_axis = np.linspace(0,1,num = recomposed_signal.shape[1])
        signal = traj
        for i in range(traj.shape[0]):
            plt.plot(x_axis,signal[i])
            plt.plot(x_axis,np.flip(recomposed_signal[2+i,:]))
            plt.legend(['truth_signal','reconstruction_signal'])
            plt.title(f"channel {i+1}")
            plt.savefig("Inv_results/reconstruction_gan_cos_comp.png")
            plt.show()

        print(np.mean(traj-traj[:,0,None]-recomposed_signal[2:,:]))

        plt.plot(L)
        plt.plot(np.flip(recomposed_signal[1,:]))
        plt.legend(['truth_length','reconstruction_length'])
        plt.show()

        plt.plot(times)
        plt.plot(np.flip(recomposed_signal[0,:]))
        plt.legend(['truth_times','reconstruction_times'])
        plt.show()

if __name__=="__main__":
    test_inverse_GAN(nb_ch, size_ts, False, time_add=True)