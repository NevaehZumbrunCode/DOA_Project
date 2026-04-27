# -------------------------
# How to Run: 
# 1) Install pandas matplotpib scikit-learn (pip install matplotlib scikit-learn)
# 2) python k-means.py
# 3) python k-meansgraphs.py
# -------------------------
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# -------------------------------
# STEP 1: show where Python is running
# -------------------------------

print("Current working directory:", os.getcwd())

# -------------------------------
# STEP 2: make folders
# -------------------------------

os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# -------------------------------
# STEP 3: load CSV
# -------------------------------

csv_path = "../results/kmeans_synthetic_results.csv"

print("Looking for CSV at:", os.path.abspath(csv_path))

if not os.path.exists(csv_path):
    print("ERROR: CSV file was not found.")
    print("Move kmeans_synthetic_results.csv into the results folder, or fix csv_path.")
    exit()

df = pd.read_csv(csv_path)

print("CSV loaded successfully.")
print(df)

# -------------------------------
# STEP 4: Runtime Graph
# -------------------------------

plt.figure()
plt.plot(df["n"], df["runtime_seconds"], marker='o', linewidth=2)
plt.title("KMeans Runtime vs Dataset Size")
plt.xlabel("Dataset Size (n)")
plt.ylabel("Runtime (seconds)")
plt.tight_layout()
runtime_path = "../graphs/kmeans_scaling_graphs/kmeans_runtime.png"
plt.savefig(runtime_path)
plt.close()
print("Saved:", os.path.abspath(runtime_path))

# -------------------------------
# STEP 5: Memory Graph
# -------------------------------

plt.figure()
plt.plot(df["n"], df["memory_mb"], marker='o', linewidth=2)
plt.title("KMeans Memory vs Dataset Size")
plt.xlabel("Dataset Size (n)")
plt.ylabel("Memory Usage (MB)")
plt.tight_layout()
memory_path = "../graphs/kmeans_scaling_graphs/kmeans_memory.png"
plt.savefig(memory_path)
plt.close()
print("Saved:", os.path.abspath(memory_path))

# -------------------------------
# STEP 6: Silhouette Graph
# -------------------------------

plt.figure()
plt.plot(df["n"], df["silhouette_score"], marker='o', linewidth=2)
plt.title("KMeans Silhouette Score vs Dataset Size")
plt.xlabel("Dataset Size (n)")
plt.ylabel("Silhouette Score")
plt.tight_layout()
silhouette_path = "../graphs/kmeans_scaling_graphs/kmeans_silhouette.png"
plt.savefig(silhouette_path)
plt.close()
print("Saved:", os.path.abspath(silhouette_path))

# -------------------------------
# STEP 7: KMeans Cluster Graph
# -------------------------------

X, _ = make_blobs(
    n_samples=2000,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

model = KMeans(n_clusters=3, random_state=42)
labels = model.fit_predict(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("KMeans Clustering on Synthetic Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
cluster_path = "../graphs/kmeans_scaling_graphs/kmeans_clusters.png"
plt.savefig(cluster_path)
plt.close()
print("Saved:", os.path.abspath(cluster_path))

print("Graphs saved successfully.")
