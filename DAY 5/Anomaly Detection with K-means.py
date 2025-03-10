import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

distances = np.min(kmeans.transform(X), axis=1)
threshold = np.percentile(distances, 95)
anomalies = X[distances > threshold]

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='black', marker='x', label='Anomalies')
plt.legend()
plt.show()
