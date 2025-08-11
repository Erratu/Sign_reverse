CRITIC_ITERS = 4  # How many critic iterations per generator iteration

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd
from scipy.spatial.distance import pdist
from signatory import signature_channels
from Algo_Seigal_inverse_path2 import SeigalAlgo

size_ts = 100
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
#Available base: "PwLinear", "BSpline", "Fourier"
base_name = "PwLinear"

class Discriminator(nn.Module):

    def __init__(self, sign_dim):

        super().__init__()
        self.sign_dim = sign_dim
        self.model = nn.Sequential(
            nn.Linear(sign_dim, 32),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.5),
            nn.Linear(32, 32),
            nn.LeakyReLU(0.2), 
            nn.Dropout(0.5),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.model(x)
    

class Generator(nn.Module):

    def __init__(self, input_dim, sign_dim):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, sign_dim),
        )

    def forward(self, x):
        out = self.model(x)
        return out 

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def standardize_per_dimension(tensor, mean, std):
    standardized = torch.where(std != 0, (tensor - mean) / std, tensor - mean)
    return standardized
        
class GAN():
    def __init__(self, epochs, batch_size, channels, input_dim, loss_function, lamb, lr_G=1e-4, lr_D=1e-4, betas=(0.0, 0.9), device="cpu"):

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.chan = channels
        self.sign_dim = signature_channels(channels, 3, scalar_term=True)
        self.input_dim = input_dim
        self.loss = loss_function
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.betas = betas
        self.lamb = lamb

        self.generator = Generator(self.input_dim,self.sign_dim).to(device)
        self.discriminator = Discriminator(self.sign_dim).to(device)

        self.discriminator.apply(weights_init)
        self.generator.apply(weights_init)

        self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=self.betas)
        self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_D, betas=self.betas)

    def train_step(self, train_data, mean, std, dir):
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )

        self.generator.train()
        self.discriminator.train()

        losses_G = []
        losses_D = []
        
        num_ep = 0
        for epoch in range(self.epochs):
            num_ep += 1
            for i, real_samples in enumerate(train_loader):
                batch_size = real_samples.shape[0]

                if (20 < epoch <= 40) or (epoch > 40 and num_ep > 10):
                    for _ in range(CRITIC_ITERS):
                        # Data for training the discriminator
                        latent_space_samples = torch.randn((batch_size, self.input_dim))
                        generated_samples = self.generator(latent_space_samples)
                        #print("generated :", generated_samples, "\nlatent :", latent_space_samples)
    
                        real_samples_norm = standardize_per_dimension(real_samples, mean, std)
    
                        # Training the discriminator
                        self.discriminator.zero_grad()
                        output_real = self.discriminator(real_samples_norm)
                        output_false = self.discriminator(generated_samples.detach())
    
                        gp = self.gp(real_samples_norm, generated_samples, self.lamb)
    
                        # 4. Compute total D loss (WGAN-GP)
                        d_loss = -output_real.mean() + output_false.mean() + gp
    
                        # 5. Optimize D
                        self.optim_D.zero_grad()
                        d_loss.backward()
                        self.optim_D.step()
    
                    # Data for training the generator
                    latent_space_samples = torch.randn((batch_size, self.input_dim))
                    # Training the generator
                    self.generator.zero_grad()
                    self.optim_G.zero_grad()
                    generated_samples = self.generator(latent_space_samples)
                    g_loss = -self.discriminator(generated_samples).mean()
                    g_loss.backward()
                    self.optim_G.step()

                    if epoch % 10 == 0 and i == 0:
                        losses_D.append(d_loss.item())
                        losses_G.append(g_loss.item())
                        print(f"Epoch: {epoch} Loss D.: {d_loss} Loss G.: {g_loss}")
                        indices = torch.randperm(train_data.size(0))[:100]
                        x_real_test = train_data[indices]
                        latent_space_samples = torch.randn((batch_size, self.input_dim))
                        x_fake_test = torch.cat([self.generator(latent_space_samples) for _ in range(10)])
                        print(self.compute_mmd(x_real_test.detach().numpy(), x_fake_test.detach().numpy()))
                        if (epoch > 40 and num_ep == 20) or epoch == 40:
                            num_ep = 0

                if epoch <= 20 or (epoch > 40 and num_ep <= 10):
                    self.optim_G.zero_grad()
                    A_comp = torch.load('Inv_results/original_A_cos_2.pt')
                    latent_space_samples = torch.randn((batch_size, self.input_dim))
                    generated_samples = self.generator(latent_space_samples) * std + mean
                    SA = SeigalAlgo(size_ts, len_base, self.chan, real_chan, depth, n_recons, size_base, time_chan=True, sig_TS=generated_samples[0].unsqueeze(0))
                    base = SA.define_base(base_name).flip([-2,-1])
                    loss_inv_ori = SA.calculate_diff(A_comp, base, par = 1, lrs = 1e-3, limits = 1e4,opt = "AdamW",eps=1e-10, params = [1,5,0,0])
                    if epoch == 0 and i == 0:
                        scale = loss_inv_ori.item()
                    else:
                        scale = 0.99 * scale + 0.01 * loss_inv_ori.item()
                    loss_inv = loss_inv_ori / (scale + 1e-8)
                    loss_inv.backward()
                    #torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.0)
                    self.optim_G.step()
                    if i == 0:
                        print(epoch, loss_inv_ori)
                        print(loss_inv)
                        total_norm = 0.0
                        for p in self.generator.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)  # norme L2 du gradient de ce paramètre
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5  # racine pour la norme L2 globale
                        print(f"Gradient norm: {total_norm:.4f}")
                        print(generated_samples[0])

        plt.plot(losses_D, "r.")
        plt.plot(losses_G, "b.")
        plt.savefig(f"./models_saved/results_{dir}.png")
        #with open(f"./resultats_num_ep_ex.txt","a") as f:
        #    f.write(f"num_epochs:{self.epochs}, num_exs:{len(train_data)//self.num_classes} : \nLoss D.: {np.mean(losses_D[self.epochs-100:])} Loss G.: {np.mean(losses_G[self.epochs-100:])} \n")
        print(f"Loss D.: {np.mean(losses_D[self.epochs-100:])} Loss G.: {np.mean(losses_G[self.epochs-100:])}")
        plt.show()
        torch.save(self.generator.state_dict(), f"./models_saved/wgan_gp/{dir}/G_model.pt")
        torch.save(self.discriminator.state_dict(), f"./models_saved/wgan_gp/{dir}/D_model.pt")

    def gp(self, real_samples, fake_samples, lambda_gp=10):
        batch_size = real_samples.size(0)

        # Génère alpha pour interpolation linéaire
        alpha = torch.rand(batch_size, 1, device=self.device)
        alpha = alpha.expand_as(real_samples)

        # Interpolation entre réel et fake
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)

        # Discriminator output
        d_interpolates = self.discriminator(interpolates)

        # Gradients de D(interpolates) par rapport à interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Norme L2 des gradients par échantillon
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)

        # Pénalité = (||grad||_2 - 1)^2
        penalty = ((gradient_norm - 1) ** 2).mean() * lambda_gp
        return penalty
    
    def rbf_kernel(self, x, y, sigma=1.0):
        """RBF kernel entre deux matrices x et y (n x d)."""
        x_norm = np.sum(x**2, axis=1).reshape(-1, 1)
        y_norm = np.sum(y**2, axis=1).reshape(1, -1)
        dist_sq = x_norm + y_norm - 2 * np.dot(x, y.T)
        return np.exp(-dist_sq / (2 * sigma**2))
    
    def compute_mmd(self, x_real, x_fake):

        sigma = np.median(pdist(np.vstack([x_real, x_fake])))

        K_XX = self.rbf_kernel(x_real, x_real, sigma)
        K_YY = self.rbf_kernel(x_fake, x_fake, sigma)
        K_XY = self.rbf_kernel(x_real, x_fake, sigma)

        mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
        return mmd
    
    