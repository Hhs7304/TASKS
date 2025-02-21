import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, seed in enumerate([0, 42, 100]):
    kmeans = KMeans(n_clusters=3, init="random", random_state=seed)
    y_kmeans = kmeans.fit_predict(X)
    axes[i].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
    axes[i].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
    axes[i].set_title(f"Random Seed: {seed}")
plt.show()
