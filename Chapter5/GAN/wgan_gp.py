CRITIC_ITERS = 2  # How many critic iterations per generator iteration

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd

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
    def __init__(self, epochs, batch_size, sign_dim, input_dim, loss_function, lamb, lr_G=1e-4, lr_D=1e-4, betas=(0.0, 0.9), device="cpu"):

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.sign_dim = sign_dim
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

        for epoch in range(self.epochs):
            for _, real_samples in enumerate(train_loader):
                batch_size = real_samples.shape[0]
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
                generated_samples = self.generator(latent_space_samples)
                g_loss = -self.discriminator(generated_samples).mean()

                self.optim_G.zero_grad()
                g_loss.backward()
                self.optim_G.step()
                #for name, param in self.generator.named_parameters():
                #    if param.grad is not None:
                #        print(f"{name} grad mean: {param.grad.mean().item()}")
        
        
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} Loss D.: {d_loss} Loss G.: {g_loss}")
                #with open("hyperparams/lambda_search.txt", 'a') as f:
                #    f.write(f"Epoch: {epoch} Loss D.: {d_loss} Loss G.: {g_loss}\n")
                
            losses_D.append(d_loss.item())
            losses_G.append(g_loss.item())

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
    