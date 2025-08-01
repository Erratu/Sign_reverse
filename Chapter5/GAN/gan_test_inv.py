import numpy as np
from Algo_Seigal_inverse_path2 import SeigalAlgo
import matplotlib.pyplot as plt
import torch
from signatory import Signature

from wgan_gp import Generator
from data_gen import data_gen_curve, create_TD

batch_size = 32
input_dim = 32
device = "cpu"
size_ts= 100

def test_inverse_GAN(size_ts, multichan, time_add=True):
    """montre le graphe de la TS associée à la signature créée par le GAN.

    Args:
        nb_ch (int): the number of data to use to train the Generator
        size_ts (int): the size of the TS to recompose
        multichan (bool): If the test data are multidimensional or not
        time_add (bool): If the data are time-augmented. Defaults to true.
    """

    signature_TS = Signature(depth = 3,scalar_term= True).to(device)
    
    times,traj = data_gen_curve(size_ts, curve=1)
    L, TS = create_TD(multichan, times, traj, time_add = True)
    np.save('Inv_results/TS_ori_1.npy', TS)
    signature = signature_TS(torch.from_numpy(TS)[None].to(device))
    sign_dim = signature.shape[1]


    data_stats = torch.load("models_saved/wgan_gp/cosine/stats_gan.pt")
    mean = data_stats['mean']
    std = data_stats['std']

    generator = Generator(input_dim, sign_dim)
    generator.load_state_dict(torch.load('models_saved/wgan_gp/cosine/G_model.pt'), strict=False)
    latent_space_samples = torch.randn((batch_size, input_dim))
    sign = generator(latent_space_samples) * std + mean
    latent_space_samples = torch.randn((batch_size, input_dim))

    print(signature, sign)

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

    A_init = torch.load('Inv_results/original_A_cos_2.pt')

    SA = SeigalAlgo(size_ts, len_base, chan, real_chan, depth, n_recons, size_base, time_chan=True, sig_TS=sign[0].unsqueeze(0), A_init=A_init)
    base = SA.define_base(base_name).flip([-2,-1])
    # Retrieve A from depth 3 signature. "par" is deprecated for the moment. If cuda is available, compute automatically from cuda.
    A = SA.retrieve_coeff_base(base, par = 1, limits = limits, lrs = lrs, opt = optim, params = params)
    torch.save(A, 'Inv_results/gan_A_cos_1.pt')
    # Recompose signal from A
    recomposed_signal = np.matmul(A.detach().numpy(),base[0].detach().numpy().T)

    SA = SeigalAlgo(size_ts, len_base, chan, real_chan, depth, n_recons, size_base, time_chan=True, sig_TS=signature, A_init=A_init)
    base = SA.define_base(base_name).flip([-2,-1])
    # Retrieve A from depth 3 signature. "par" is deprecated for the moment. If cuda is available, compute automatically from cuda.
    A_original = SA.retrieve_coeff_base(base, par = 1, limits = limits, lrs = lrs, opt = optim, params = params)
    torch.save(A_original, 'Inv_results/original_A_cos_2.pt')
    # Recompose signal from A
    recomposed_signal_original = np.matmul(A_original.detach().numpy(),base[0].detach().numpy().T)
    
    #See the result

    if not multichan:
        if time_add:
            i = 1
        else:
            i=0
        x_axis = np.linspace(0,10,num = recomposed_signal.shape[1])
        plt.plot(x_axis,traj-traj[0])
        plt.plot(x_axis,np.flip(recomposed_signal[i+1,:]))
        plt.plot(x_axis,np.flip(recomposed_signal_original[i+1,:]))
        plt.legend(['truth_signal','GAN_recon_signal','reconstruction_signal'])
        plt.savefig("Inv_results/reconstruction_cos_pw1.png")
        plt.show()
        print(np.mean(traj-traj[0]-recomposed_signal[i+1,:]))

        plt.plot(x_axis,L)
        plt.plot(x_axis,np.flip(recomposed_signal[1,:]))
        plt.plot(x_axis,np.flip(recomposed_signal_original[1,:]))
        plt.legend(['truth_length','GAN_length','reconstruction_length'])
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
    test_inverse_GAN(size_ts, False, time_add=True)