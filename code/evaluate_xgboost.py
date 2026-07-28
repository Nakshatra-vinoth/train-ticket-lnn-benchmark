from pathlib import Path
import time
import json

import numpy as np
import xgboost as xgb

DATA_DIR = Path(__file__).resolve().parent.parent / "benchmark"

# Load data
X_test = np.load(DATA_DIR / "X_test_scaled.npy")

def summarize_window(X):
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    last = X[:, -1, :]
    return np.concatenate([mean, std, mn, mx, last], axis=1)

X_test_flat = summarize_window(X_test)

# Load model
model = xgb.Booster()
model.load_model("xgboost_baseline.json")

print(f"Trees: {model.num_boosted_rounds()}")

dtest = xgb.DMatrix(X_test_flat)

# Warm-up
for _ in range(20):
    model.predict(dtest)

N = 100

start = time.perf_counter()

for _ in range(N):
    model.predict(dtest)

end = time.perf_counter()

latency_ms = ((end - start) * 1000) / (N * len(X_test))

print(f"Average inference latency: {latency_ms:.6f} ms/sample")