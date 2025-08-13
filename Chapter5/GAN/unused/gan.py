import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Discriminator(nn.Module):

    def __init__(self, sign_dim):

        super().__init__()
        self.sign_dim = sign_dim
        self.model = nn.Sequential(
            nn.Linear(sign_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)
    

class Generator(nn.Module):

    def __init__(self, input_dim, sign_dim):

        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, sign_dim),
        )

    def forward(self, x):
        out = self.model(x)
        return out 

    
class GAN():
    def __init__(self, epochs, batch_size, sign_dim, input_dim, loss_function, lr_G, lr_D, betas=(0.5, 0.999), device="cpu"):

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.sign_dim = sign_dim
        self.input_dim = input_dim
        self.loss = loss_function
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.betas = betas

        self.generator = Generator(self.input_dim,self.sign_dim).to(device)
        self.discriminator = Discriminator(self.sign_dim).to(device)

        self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=self.betas)
        self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_D, betas=self.betas)

    def train_step(self, train_data, G_file, D_file):

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

                # Data for training the discriminator
                real_samples_labels = torch.ones((batch_size, 1))
                latent_space_samples = torch.randn((batch_size, self.input_dim))

                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros((batch_size, 1))

                # Training the discriminator
                self.discriminator.zero_grad()
                output_real = self.discriminator(real_samples)
                output_false = self.discriminator(generated_samples.detach())
                loss_discriminator = (self.loss(
                    output_real, real_samples_labels) + self.loss(
                    output_false, generated_samples_labels)) / 2
                loss_discriminator.backward()
                self.optim_D.step()

                # Data for training the generator
                latent_space_samples = torch.randn((batch_size, self.input_dim))

                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                loss_generator = self.loss(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                self.optim_G.step()

                #for name, param in self.generator.named_parameters():
                #    if param.grad is not None:
                #        print(f"{name} grad mean: {param.grad.mean().item()}")
        
        
            if epoch % 10 == 0:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator} Loss G.: {loss_generator}")
                
            losses_D.append(loss_discriminator.item())
            losses_G.append(loss_generator.item())

        plt.plot(losses_D, "r.")
        plt.plot(losses_G, "b.")
        plt.savefig(f"./models_saved/results_{G_file}.png")
        #with open(f"./resultats_num_ep_ex.txt","a") as f:
        #    f.write(f"num_epochs:{self.epochs}, num_exs:{len(train_data)//self.num_classes} : \nLoss D.: {np.mean(losses_D[self.epochs-100:])} Loss G.: {np.mean(losses_G[self.epochs-100:])} \n")
        print(f"Loss D.: {np.mean(losses_D[self.epochs-100:])} Loss G.: {np.mean(losses_G[self.epochs-100:])}")
        plt.show()
        torch.save(self.generator.state_dict(), f"./models_saved/{G_file}.pt")
        torch.save(self.discriminator.state_dict(), f"./models_saved/{D_file}.pt")

