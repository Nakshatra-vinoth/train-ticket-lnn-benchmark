from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "benchmark"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# ----------------------------
# Load test data
# ----------------------------
X_test = np.load(DATA_DIR / "X_test_scaled.npy")

device = torch.device("cpu")
torch.set_num_threads(1)

# ==========================================================
# PASTE THE LSTMBaseline CLASS HERE
# (Copy only the class definition from train_lstm_baseline.py)
# ==========================================================
class GRUBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, h_n = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


# ==========================================================

model = GRUBaseline(input_dim=X_test.shape[2]).to(device)

model.load_state_dict(
    torch.load(
        MODEL_DIR / "gru_baseline_best.pt",
        map_location=device,
    )
)

model.eval()

# ----------------------------
# Parameter count
# ----------------------------
params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Trainable parameters: {params:,}")

# ----------------------------
# Prepare test tensor
# ----------------------------
test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
).to(device)

# ----------------------------
# Warm-up
# ----------------------------
with torch.no_grad():
    for _ in range(20):
        _ = model(test_tensor)

# ----------------------------
# Measure inference latency
# ----------------------------
N = 100

import gc
gc.collect()

start = time.perf_counter()

with torch.no_grad():
    for _ in range(N):
        _ = model(test_tensor)

end = time.perf_counter()

total_samples = N * len(X_test)

latency_ms = ((end - start) * 1000) / total_samples

print(f"Average inference latency: {latency_ms:.6f} ms/sample")