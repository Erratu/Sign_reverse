import torch
from torch import nn
import numpy as np
import torchvision.utils as vutils

class Discriminator(nn.Module):

    def __init__(self, classes, length):

        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(classes, classes)
        self.model = nn.Sequential(
            nn.Linear(classes + length, 256, False, True),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128, True, True),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64, True, True),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32, False, False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1, False, False),
            nn.Sigmoid(),
        )

    def forward(self, x, labels):
        x = torch.cat((x, self.label_embedding(labels)), -1)
        return self.model(x)
    
class Generator(nn.Module):

    def __init__(self, classes, length, input_dim):

        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(classes, classes)
        self.model = nn.Sequential(
            nn.Linear(input_dim + classes, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, classes + length),
        )

    def forward(self, x, labels, dim):
        x = torch.cat((x, self.label_embedding(labels)), -1)
        out = self.model(x)
        return out.view(-1, dim[0], dim[1])

    
class GAN():
    def __init__(self, epochs, batch_size, num_classes, sign_shape, input_dim, loss_G, loss_D, lr, device="cpu"):

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.shape = sign_shape
        self.loss_G = loss_G
        self.loss_D = loss_D
        self.generated_data = []
        self.length = self.shape[0]*self.shape[1]

        self.generator = Generator(self.num_classes, self.length, self.input_dim).to(device)
        self.discriminator = Discriminator(self.num_classes, self.length).to(device)

        self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

    def train_step(self, train_data, classes):

        train_set = [
            (train_data[i], classes[i]) for i in range(len(classes))
        ]

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )

        self.generator.train()
        self.discriminator.train()
        viz_z = torch.zeros((batch_size, self.input_dim), device=self.device)
        viz_noise = torch.randn(batch_size, self.input_dim, device=self.device)
        nrows = batch_size // 8
        viz_label = torch.LongTensor(np.array([num for _ in range(nrows) for num in range(8)])).to(self.device)

        for epoch in range(self.epochs):
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(self.device), target.to(self.device)
                batch_size = data.size(0)
                real_label = torch.full((batch_size, 1), 1., device=self.device)
                fake_label = torch.full((batch_size, 1), 0., device=self.device)

                # Train G
                self.generator.zero_grad()
                z_noise = torch.randn(batch_size, self.input_dim, device=self.device)
                #x_fake_labels = torch.randint(0, self.classes, (batch_size,), device=self.device)
                x_fake_labels = torch.randint(1, 2, (batch_size,), device=self.device)
                x_fake = self.generator(z_noise, x_fake_labels)
                y_fake_g = self.discriminator(x_fake, x_fake_labels)
                g_loss = self.discriminator.loss(y_fake_g, real_label)
                g_loss.backward()
                self.optim_G.step()

                # Train D
                self.discriminator.zero_grad()
                y_real = self.discriminator(data, target)
                d_real_loss = self.discriminator.loss(y_real, real_label)
                y_fake_d = self.discriminator(x_fake.detach(), x_fake_labels)
                d_fake_loss = self.discriminator.loss(y_fake_d, fake_label)
                d_loss = (d_real_loss + d_fake_loss) / 2
                d_loss.backward()
                self.optim_D.step()

                if batch_idx % self.lr == 0 and batch_idx > 0:
                    print('Epoch {} [{}/{}] loss_D: {:.4f} loss_G: {:.4f}'.format(
                                epoch, batch_idx, len(train_loader),
                                d_loss.mean().item(),
                                g_loss.mean().item()))

                    with torch.no_grad():
                        viz_sample = self.generator(viz_noise, viz_label)
                        self.generated_data.append(vutils.make_grid(viz_sample, normalize=True))

        