from pathlib import Path
import time
import gc

import numpy as np
import torch
import torch.nn as nn
from ncps.torch import LTC

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

# Use only first 100 samples for benchmarking
test_tensor = test_tensor[:100]
N = 20
# ----------------------------
# Model
# ----------------------------
class LTCBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, ode_unfolds=3):
        super().__init__()
        self.ltc = LTC(
            input_dim,
            hidden_dim,
            batch_first=True,
            ode_unfolds=ode_unfolds,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.ltc(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


model = LTCBaseline(input_dim=X_test.shape[2]).to(device)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "ltc_baseline_best.pt",
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

print("Starting one forward pass...")

start = time.perf_counter()

with torch.no_grad():
    _ = model(test_tensor)

end = time.perf_counter()

elapsed = end - start

print(f"100 samples took {elapsed:.3f} seconds")

latency_ms = (elapsed * 1000) / len(test_tensor)

print(f"Average inference latency: {latency_ms:.6f} ms/sample")