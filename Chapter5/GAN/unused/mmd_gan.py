import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

    
class MMDGAN():
    def __init__(self, epochs, batch_size, num_classes, class_sizes, input_dim, loss_function, lr_G, sigma=1.0, betas=(0.5, 0.999), device="cpu"):

        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.sign_dim = max(class_sizes)
        self.loss = loss_function
        self.lr_G = lr_G
        self.betas = betas
        self.class_sizes = class_sizes
        self.sigma=sigma

        self.generator = Generator(self.input_dim,self.sign_dim, self.num_classes).to(device)

        self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr_G, betas=self.betas)

    def train_step(self, train_data):

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True
        )

        self.generator.train()
        losses_G = []
        
        for epoch in range(self.epochs):
            for _, (real_samples, classes_real) in enumerate(train_loader):
                batch_size = real_samples.shape[0]

                # Data for training the generator
                real_samples_labels = torch.ones((batch_size, 1))
                latent_space_samples = torch.randn((batch_size, self.input_dim))

                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples, classes_real)

                list_mmd = []
                for i in range(self.num_classes):
                    class_size = self.class_sizes[i]
                    idx_class = torch.nonzero(classes_real == i, as_tuple=False).squeeze()
                    generated_1c = generated_samples[idx_class,:class_size]
                    real_1c = real_samples[idx_class,:class_size]

                    if class_size > 100:
                        real_np = real_1c.detach().cpu().numpy()
                        generated_np = generated_1c.detach().cpu().numpy()

                        pca_full = PCA(n_components=min(idx_class.shape[0], class_size))
                        pca_full.fit(real_np)
                        cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
                        target_dim = np.searchsorted(cumulative_variance, 0.95) + 1
                        #print(f"Dimension retenue : {target_dim}")
                        pca = PCA(n_components=target_dim)
                        real_np = pca.fit_transform(real_np)
                        generated_np = pca.fit_transform(generated_np)

                        real_1c = torch.from_numpy(real_np).to(real_1c.device)  
                        generated_1c = torch.from_numpy(generated_np).to(generated_1c.device)  


                    list_mmd.append(self.compute_mmd(generated_1c, real_1c, self.sigma))

                loss_generator = torch.stack(list_mmd).mean()
                loss_generator.backward()
                self.optim_G.step()

                #for name, param in self.generator.named_parameters():
                #    if param.grad is not None:
                #        print(f"{name} grad mean: {param.grad.mean().item()}")
        
        
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss G.: {loss_generator}")
                
            losses_G.append(loss_generator.item())

        plt.plot(losses_G, "b.")
        plt.savefig(f"./models_saved/results_mmd_sign_gen.png")
        #with open(f"./resultats_num_ep_ex.txt","a") as f:
        #    f.write(f"num_epochs:{self.epochs}, num_exs:{len(train_data)//self.num_classes} : \nLoss G.: {np.mean(losses_G[self.epochs-100:])} \n")
        print(f"Loss G.: {np.mean(losses_G[self.epochs-100:])}")
        plt.show()
        torch.save(self.generator.state_dict(), "./models_saved/generator_sign_mmd.pt")

    def gaussian_kernel(self, x, y, sigma):
        # x, y: (batch_size, dim)
        x_expanded = x.unsqueeze(1)  # (batch_size, 1, dim)
        y_expanded = y.unsqueeze(0)  # (1, batch_size, dim)
        dist = ((x_expanded - y_expanded)**2).sum(2)
        return torch.exp(-dist / (2 * sigma**2))

    def compute_mmd(self, x, y, sigma):
        Kxx = self.gaussian_kernel(x, x, sigma)
        Kyy = self.gaussian_kernel(y, y, sigma)
        Kxy = self.gaussian_kernel(x, y, sigma)
        mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
        return mmd
