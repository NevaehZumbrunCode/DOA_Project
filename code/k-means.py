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
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.makedirs("../results", exist_ok=True)

sizes = [1000, 5000, 10000, 20000]
results = []

for n in sizes:
    X, _ = make_blobs(
        n_samples=n,
        centers=3,
        cluster_std=1.0,
        random_state=42
    )

    # warm-up run: do NOT record this
    warmup_model = KMeans(n_clusters=3, random_state=42, n_init=10)
    warmup_model.fit_predict(X)

    runtimes = []
    memories = []
    final_labels = None

    # repeat and average
    for trial in range(10):
        tracemalloc.start()
        start_time = time.perf_counter()

        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = model.fit_predict(X)

        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        runtimes.append(end_time - start_time)
        memories.append(peak / (1024 * 1024))
        final_labels = labels

    runtime = sum(runtimes) / len(runtimes)
    memory_mb = sum(memories) / len(memories)
    sil = silhouette_score(X, final_labels)

    results.append({
        "algorithm": "KMeans",
        "dataset": "Synthetic",
        "n": n,
        "k": 3,
        "runtime_seconds": runtime,
        "memory_mb": memory_mb,
        "silhouette_score": sil
    })

df = pd.DataFrame(results)
df.to_csv("../results/kmeans_synthetic_results.csv", index=False)

print(df)
print("Saved successfully.")
