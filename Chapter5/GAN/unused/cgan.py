import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Discriminator(nn.Module):

    def __init__(self, sign_dim, num_classes):

        super().__init__()
        self.sign_dim = sign_dim
        self.label_embedding = nn.Embedding(num_classes, num_classes//2)
        self.model = nn.Sequential(
            nn.Linear(sign_dim + num_classes//2, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        x = torch.cat((x, self.label_embedding(labels)), -1)
        return self.model(x)
    

class Generator(nn.Module):

    def __init__(self, input_dim, sign_dim, num_classes):

        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes//2)
        self.model = nn.Sequential(
            nn.Linear(input_dim + num_classes//2, 64),
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

    def forward(self, x, labels):
        x = torch.cat((x, self.label_embedding(labels)), -1)
        out = self.model(x)
        return out 

    
class GAN():
    def __init__(self, epochs, batch_size, num_classes, sign_dim, input_dim, loss_function, lr_G, lr_D, betas=(0.5, 0.999), device="cpu"):

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.sign_dim = max(sign_dim)
        self.loss = loss_function
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.betas = betas

        self.generator = Generator(self.input_dim,self.sign_dim, self.num_classes).to(device)
        self.discriminator = Discriminator(self.sign_dim, self.num_classes).to(device)

        self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=self.betas)
        self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr_D, betas=self.betas)

    def train_step(self, train_data):

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )

        self.generator.train()
        self.discriminator.train()

        losses_G = []
        losses_D = []
        
        for epoch in range(self.epochs):
            for _, (real_samples, classes_real) in enumerate(train_loader):
                batch_size = real_samples.shape[0]
                # Data for training the discriminator
                real_samples_labels = torch.ones((batch_size, 1))
                latent_space_samples = torch.randn((batch_size, self.input_dim))

                classes_gen = torch.randint(0,self.num_classes,(batch_size,))
                generated_samples = self.generator(latent_space_samples, classes_gen)

                generated_samples_labels = torch.zeros((batch_size, 1))
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )
                classes_all = torch.cat((classes_real, classes_gen))

                # Training the discriminator -> modifier pour entrainer sur vrais puis sur faux, D ne sort que oui ou non
                self.discriminator.zero_grad()
                output_discriminator = self.discriminator(all_samples, classes_all)
                loss_discriminator = self.loss(
                    output_discriminator, all_samples_labels)
                loss_discriminator.backward()
                self.optim_D.step()

                # Data for training the generator
                latent_space_samples = torch.randn((batch_size, self.input_dim))
                classes_gen = torch.randint(0,self.num_classes,size=(batch_size,))

                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples, classes_gen)
                output_discriminator_generated = self.discriminator(generated_samples, classes_gen)
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
        plt.savefig(f"./models_saved/results_sign_gen.png")
        #with open(f"./resultats_num_ep_ex.txt","a") as f:
        #    f.write(f"num_epochs:{self.epochs}, num_exs:{len(train_data)//self.num_classes} : \nLoss D.: {np.mean(losses_D[self.epochs-100:])} Loss G.: {np.mean(losses_G[self.epochs-100:])} \n")
        print(f"Loss D.: {np.mean(losses_D[self.epochs-100:])} Loss G.: {np.mean(losses_G[self.epochs-100:])}")
        plt.show()
        torch.save(self.generator.state_dict(), "./models_saved/generator_sign_gen.pt")
        torch.save(self.discriminator.state_dict(), "./models_saved/discriminator_sign_gen.pt")

