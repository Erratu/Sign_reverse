import numpy as np
import matplotlib.pylab as plt
import torch

# Dummy model param
model = torch.nn.Linear(1, 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

# CosineAnnealingWarmRestarts scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=2000, T_mult=1, eta_min=1e-5
)

# Simuler les valeurs de LR à chaque step
lr_values = []
num_steps = 12000  # Total d'étapes à simuler

for step in range(num_steps):
    scheduler.step(step)
    lr = optimizer.param_groups[0]['lr']
    lr_values.append((step, lr))

# Conversion en array pour affichage
lr_values = np.array(lr_values)

# Affichage
plt.figure(figsize=(8, 4))
plt.plot(lr_values[:, 0], lr_values[:, 1], label="LR (CosineWarmRestarts)")
plt.xlabel("Step")
plt.ylabel("Learning rate")
plt.title("CosineAnnealingWarmRestarts")
plt.grid(True)
plt.xlim([0, num_steps])
plt.ylim(bottom=0)
plt.legend()
plt.tight_layout()
plt.show()

plt.hist(variable, bins=50)
plt.title("Distribution de la variable")
plt.xlabel("Valeur")
plt.ylabel("Fréquence")
plt.show()