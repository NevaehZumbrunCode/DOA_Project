# -------------------------
# How to Run: 
# 1) Install pandas matplotpib scikit-learn (pip install matplotlib scikit-learn)
# 2) python k-means.py
# 3) python k-meansgraphs.py
# -------------------------

import os
import time
import tracemalloc
import pandas as pd
from sklearn.datasets import make_blobs, load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------------
# STEP 1: folders
# -----------------------------------

os.makedirs("results", exist_ok=True)

# -----------------------------------
# STEP 2: dataset sizes
# -----------------------------------

sizes = [200, 500, 1000, 2000]

results = []

# -----------------------------------
# STEP 3: synthetic dataset experiments
# -----------------------------------

for n in sizes:
    X, _ = make_blobs(
        n_samples=n,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )

    tracemalloc.start()
    start_time = time.time()

    model = KMeans(
        n_clusters=3,
        random_state=42
    )

    labels = model.fit_predict(X)

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    runtime = end_time - start_time
    memory_mb = peak / (1024 * 1024)
    sil = silhouette_score(X, labels)

    results.append({
        "algorithm": "KMeans",
        "dataset": "Synthetic",
        "n": n,
        "k": 3,
        "runtime_seconds": runtime,
        "memory_mb": memory_mb,
        "silhouette_score": sil
    })

# -----------------------------------
# STEP 4: save CSV
# -----------------------------------

df = pd.DataFrame(results)
df.to_csv("../results/kmeans_synthetic_results.csv", index=False)

print(df)
print("Saved successfully.")
