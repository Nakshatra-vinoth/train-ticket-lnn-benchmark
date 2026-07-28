from pathlib import Path
import time
import gc

import numpy as np
import torch
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "benchmark"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# ----------------------------
# Settings
# ----------------------------
torch.set_num_threads(1)
device = torch.device("cpu")

# ----------------------------
# Load test data
# ----------------------------
X_test = np.load(DATA_DIR / "X_test_scaled.npy")

test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
).to(device)

# ----------------------------
# Build model
# ----------------------------
INPUT_SIZE = X_test.shape[2]
UNITS = 64

wiring = AutoNCP(UNITS, 1)
model = CfC(INPUT_SIZE, wiring, batch_first=True).to(device)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "cfc_baseline_best.pt",
        map_location=device,
    )
)

model.eval()

# ----------------------------
# Parameter count
# ----------------------------
params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print(f"Trainable parameters: {params:,}")

# ----------------------------
# Warm-up
# ----------------------------
with torch.no_grad():
    for _ in range(20):
        _ = model(test_tensor)

# ----------------------------
# Measure inference latency
# ----------------------------
gc.collect()

N = 100

start = time.perf_counter()

with torch.no_grad():
    for _ in range(N):
        _ = model(test_tensor)

end = time.perf_counter()

latency_ms = ((end - start) * 1000) / (N * len(X_test))

print(f"Average inference latency: {latency_ms:.6f} ms/sample")