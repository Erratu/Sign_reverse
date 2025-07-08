import torch
from torch import nn
import torchvision.utils as vutils

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()
        #self.label_embedding = nn.Embedding(classes, classes//2)
        self.model = nn.Sequential(
            #nn.Linear(classes//2 + length, 128),
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        #x = torch.cat((x, self.label_embedding(labels)), -1)
        return self.model(x)
    

class Generator(nn.Module):

    def __init__(self):

        super().__init__()
        #self.label_embedding = nn.Embedding(classes, classes//2)
        #self.T=T
        #self.D=D
        self.model = nn.Sequential(
            #nn.Linear(input_dim + classes//2, 64),
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        #x = torch.cat((x, self.label_embedding(labels)), -1)
        out = self.model(x)
        return out#.view(-1, self.T, self.D) 

    
class GAN():
    def __init__(self, epochs, batch_size, num_classes, sign_shape, input_dim, loss_function, lr, log_interval, device="cpu"):

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.shape = sign_shape
        self.loss = loss_function
        self.generated_data = []
        self.lr = lr
        self.log_interval = log_interval

        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)

        self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

    def train_step(self, train_data):

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )


        self.generator.train()
        self.discriminator.train()
        for epoch in range(self.epochs):
            for n, (real_samples, _) in enumerate(train_loader):

                # Data for training the discriminator
                real_samples_labels = torch.ones((self.batch_size, 1))
                latent_space_samples = torch.randn((self.batch_size, 2))
                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros((self.batch_size, 1))
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )

                # Training the discriminator
                self.discriminator.zero_grad()
                output_discriminator = self.discriminator(all_samples)
                loss_discriminator = self.loss(
                    output_discriminator, all_samples_labels)
                loss_discriminator.backward()
                self.optim_D.step()

                # Data for training the generator
                latent_space_samples = torch.randn((self.batch_size, 2))

                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                loss_generator = self.loss(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                self.optim_G.step()


                # Show loss
                if epoch % 10 == 0 and n == self.batch_size - 1:
                    print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                    print(f"Epoch: {epoch} Loss G.: {loss_generator}")

                for name, param in self.generator.named_parameters():
                    if param.grad is not None:
                        print(f"{name} grad mean: {param.grad.mean().item()}")

        torch.save(self.generator.state_dict(), "./models_saved/generator_sin.pt")