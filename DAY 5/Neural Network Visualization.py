import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

model = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 1), nn.Sigmoid())

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

xx, yy = np.meshgrid(np.linspace(-2, 3, 100), np.linspace(-1.5, 2, 100))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
preds = model(grid).detach().numpy().reshape(xx.shape)

plt.contourf(xx, yy, preds, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y[:, 0], cmap='coolwarm')
plt.show()
