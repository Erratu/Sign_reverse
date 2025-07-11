from torch import nn
from Chapter5.GAN.unused.cgan import GAN
from Chapter5.GAN.unused.mmd_gan import MMDGAN
from data_gen import create_training_data_cgan

# Define variables
CUDA = False
batch_size = 64
lr_G = 0.001
lr_D = 0.0002
num_epochs = 1000
loss_function = nn.BCELoss()
num_classes = 6 # type des données générées
input_dim = 8
nb_ch = 1000

num_ex = 400
channels = [3,3,3,4,3,8]
gen_size = [int((channel*(channel**3 -1))/(channel-1)) for channel in channels]

mmd_batch_size = 256
mmd_num_ex = 1000
mmd_num_epochs = 200

def grid_search(train_set):
    # Learning rates
    lrG_values = [1e-4, 2e-4, 5e-4, 1e-3]
    lrD_values = [1e-4, 2e-4, 5e-4, 1e-3]
    # Betas pour Adam
    betas_values = [(0.5, 0.999), (0.4, 0.9), (0.0, 0.9)]
    # Latent space (z) dimension à tester en dernier, d'abord 16 puis 32 puis 64 puis 8
    latent_dim_values = [8, 16, 32, 64]
    # Optionnel : batch size
    batch_size_values = [16, 32, 64, 128]
    size = 16
    for dim in latent_dim_values:
        for beta in betas_values:
            for lrG in lrG_values:
                for lrD in lrD_values:
                    gan = GAN(200, size, num_classes, gen_size, dim, loss_function, lrG, lrD, beta)
                    gan.train_step(train_set)

def grid_search2(train_set):
    # Learning rates
    lrG_values = [5e-4, 1e-3]
    lrD_values = [2e-4, 1e-3]
    latent_dim_values = [8, 16]
    batch_size_values = [32, 64, 128]
    for dim in latent_dim_values:
        for size in batch_size_values:
            for lrG in lrG_values:
                for lrD in lrD_values:
                    gan = GAN(400, size, num_classes, gen_size, dim, loss_function, lrG, lrD)
                    gan.train_step(train_set)

def grid_search_hyperhyper_cgan():
    for num_ex in [200,300,400,500,600,700]:
        train_data = create_training_data_cgan([num_ex for _ in range(num_classes)], gen_size)
        for num_e in [100,200,300,400,500,600]:
            gan = GAN(num_e, batch_size, num_classes, gen_size, input_dim, loss_function, lr_G, lr_D)
            gan.train_step(train_data)