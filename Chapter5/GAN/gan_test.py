from torch import nn
from gan import GAN
#from mmd_gan import MMDGAN
from data_gen import create_training_data_gan
from wgan_gp import GAN as WGAN

# Define variables
CUDA = False
batch_size = 32
lr_G = 1e-4
lr_D = 1e-4
num_epochs = 300
loss_function = nn.BCEWithLogitsLoss()
input_dim = 8
nb_ch = 2000
channels = [3,3,3,4,3,7]

distr_num = 5
channel = channels[distr_num]
sign_dim = int((channel*(channel**3 -1))/(channel-1))

#num_classes = 6 # type des données générées
#num_ex = 400
#gen_size = [int((channel*(channel**3 -1))/(channel-1)) for channel in channels]

#mmd_batch_size = 256
#mmd_num_ex = 1000
#mmd_num_epochs = 200

def test_inverse_GAN(TS, traj, times, size_ts, multichan, time_add):
    chan = TS.shape[1]
    len_base = size_ts-1
    SA = SeigalAlgo(TS, len_base, chan, real_chan = 0, depth = 3, n_recons = 2, size_base = len_base+1,time_chan = True)

    #Available base: "PwLinear", "BSpline", "Fourier"
    base_name = "PwLinear"
    base = SA.define_base(base_name).flip([-2,-1])

    #Number of iteration for optimisation
    limits = 20000
    #Learning rate (there is a patience schedule)
    lrs = 1e-2

    #Available optimizers : "Adam", "AdamW" and "LBFGS"
    optim = "AdamW"
    # params are [lambda_length, lambda_frontier, lambda_levy, lambda_ridge]
    # [1,5,0,1] , [5,1,0,1] or the same with lambda_ridge = 0 worked well
    params = [1,5,0,0]
    pack = "torch"

    # Retrieve A from depth 3 signature. "par" is deprecated for the moment. If cuda is available, compute automatically from cuda.
    A = SA.retrieve_coeff_base(base, par = 1, limits = limits, lrs = lrs, opt = optim, params = params)
    # Recompose signal from A
    recomposed_signal = np.matmul(A.detach().numpy(),base[0].detach().numpy().T)

    #See the result

    if not multichan:
        if time_add:
            i = 1
        else:
            i=0
        x_axis = np.linspace(0,1,num = recomposed_signal.shape[1])
        plt.plot(x_axis,traj-traj[0])
        plt.plot(x_axis,np.flip(recomposed_signal[i+1,:]))
        plt.legend(['truth_signal','reconstruction_signal'])
        plt.show()
        print(np.mean(traj-traj[0]-recomposed_signal[i+1,:]))

        plt.plot(x_axis,L)
        plt.plot(x_axis,np.flip(recomposed_signal[1,:]))
        plt.legend(['truth_length','reconstruction_length'])
        plt.show()

    else:
        x_axis = np.linspace(0,1,num = recomposed_signal.shape[1])
        signal = traj-traj[:,0,None]
        for i in range(traj.shape[0]):
            plt.plot(x_axis,signal[i])
            plt.plot(x_axis,np.flip(recomposed_signal[2+i,:]))
            plt.legend(['truth_signal','reconstruction_signal'])
            plt.title(f"channel {i+1}")
            plt.show()

        print(np.mean(traj-traj[:,0,None]-recomposed_signal[2:,:]))


        plt.plot(signal[0],signal[1])
        plt.plot(np.flip(recomposed_signal[2,:]),np.flip(recomposed_signal[3,:]))
        plt.legend(['truth_signal','reconstruction_signal'])
        plt.show()

        plt.plot(L)
        plt.plot(np.flip(recomposed_signal[1,:]))
        plt.legend(['truth_length','reconstruction_length'])
        plt.show()

        plt.plot(times)
        plt.plot(np.flip(recomposed_signal[0,:]))
        plt.legend(['truth_times','reconstruction_times'])
        plt.show()




if __name__ == "__main__":
    train_data = create_training_data_gan(nb_ch, distr_num)
    wgan = WGAN(num_epochs, batch_size, sign_dim, input_dim, loss_function, lr_G, lr_D)
    wgan.train_step(train_data,"poly_G_model", "poly_D_model")





