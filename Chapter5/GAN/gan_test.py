from torch import nn
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
input_dim = 32
nb_ch = 2000
channels = [3,3,3,4,3,7]

distr_num = 0
channel = channels[distr_num]
sign_dim = int((channel*(channel**3 -1))/(channel-1))

#num_classes = 6 # type des données générées
#num_ex = 400
#gen_size = [int((channel*(channel**3 -1))/(channel-1)) for channel in channels]

#mmd_batch_size = 256
#mmd_num_ex = 1000
#mmd_num_epochs = 200


if __name__ == "__main__":
    train_data = create_training_data_gan(nb_ch, distr_num)
    wgan = WGAN(num_epochs, batch_size, sign_dim, input_dim, loss_function, 10, lr_G, lr_D)
    wgan.train_step(train_data,"poly_G_model", "poly_D_model")



