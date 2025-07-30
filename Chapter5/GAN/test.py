import matplotlib.pyplot as plt
import numpy as np
from data_gen import create_cosine, create_TD

size_ts = 600

recomposed_signal = np.load('Inv_results/original_A_1.npy')
times,traj = create_cosine(size_ts)
L, TS = create_TD(False, times, traj, time_add = True)

x_axis = np.linspace(0,10,num = recomposed_signal.shape[1])
plt.plot(x_axis,traj-traj[0])
plt.plot(x_axis,np.flip(recomposed_signal[2,:]))
#plt.plot(x_axis,np.flip(recomposed_signal[i+1,:]))
plt.legend(['truth_signal','reconstruction_signal','GAN_recon_signal'])
plt.savefig("Inv_results/reconstruction_cos_pw.png")
plt.show()
plt.plot(x_axis,L)
plt.plot(x_axis,np.flip(recomposed_signal[1,:]))
plt.legend(['truth_length','reconstruction_length'])
plt.show()