from torch import nn
from gan import GAN
#from mmd_gan import MMDGAN
from data_gen import create_training_data_gan

# Define variables
CUDA = False
batch_size = 64
lr_G = 0.0001
lr_D = 0.0001
num_epochs = 300
loss_function = nn.BCEWithLogitsLoss()
input_dim = 8
nb_ch = 2000
channels = [3,3,3,4,3,8]

distr_num = 0
channel = channels[distr_num]
sign_dim = int((channel*(channel**3 -1))/(channel-1))

#num_classes = 6 # type des données générées
# num_ex = 400
#channels = [3,3,3,4,3,8]
#gen_size = [int((channel*(channel**3 -1))/(channel-1)) for channel in channels]

#mmd_batch_size = 256
#mmd_num_ex = 1000
#mmd_num_epochs = 200


if __name__ == "__main__":
    train_data = create_training_data_gan(nb_ch, distr_num)
    gan = GAN(num_epochs, batch_size, sign_dim, input_dim, loss_function, lr_G, lr_D)
    gan.train_step(train_data,"poly_G_model", "poly_D_model")





