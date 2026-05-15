
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("Mall_Customers.csv")
print("Dataset shape:", df.shape)
print(df.head())


X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(K_range, inertias, "bo-")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("elbow_plot.png", dpi=150, bbox_inches="tight")
plt.show()


kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)


plt.figure(figsize=(8, 6))
for i in range(5):
    cluster = df[df["Cluster"] == i]
    plt.scatter(cluster["Annual Income (k$)"], cluster["Spending Score (1-100)"], label=f"Cluster {i}")

centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], s=200, c="black", marker="X", label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments (K-Means Clustering)")
plt.legend()
plt.savefig("clusters_plot.png", dpi=150, bbox_inches="tight")
plt.show()


print("\nCluster Summary:")
print(df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1))
