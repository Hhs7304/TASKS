from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

kmeans_euclidean = KMeans(n_clusters=3, metric="euclidean", random_state=42)
kmeans_euclidean.fit(X)

kmeans_manhattan = KMeans(n_clusters=3, metric="manhattan", random_state=42)
kmeans_manhattan.fit(X)

print("Silhouette Score (Euclidean):", silhouette_score(X, kmeans_euclidean.labels_))
print("Silhouette Score (Manhattan):", silhouette_score(X, kmeans_manhattan.labels_))
