import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from ncps.torch import LTC
from ncps.wirings import FullyConnected

torch.manual_seed(42)

X_train = np.load("X_train_scaled.npy")
X_val = np.load("X_val_scaled.npy")
X_test = np.load("X_test_scaled.npy")
y_train = np.load("y_train_scaled.npy")
y_val = np.load("y_val_scaled.npy")
y_test = np.load("y_test_scaled.npy")

with open("scaler_params.json") as f:
    scaler = json.load(f)
y_log_mean, y_log_std = scaler["y_log_mean"], scaler["y_log_std"]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

device = torch.device("cpu")

def to_loader(X, y, batch_size=64, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

train_loader = to_loader(X_train, y_train, shuffle=True)
val_loader = to_loader(X_val, y_val, shuffle=False)
test_loader = to_loader(X_test, y_test, shuffle=False)

class LTCBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, ode_unfolds=3):
        super().__init__()
        self.ltc = LTC(input_dim, hidden_dim, batch_first=True, ode_unfolds=ode_unfolds)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, h_n = self.ltc(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden).squeeze(-1)

model = LTCBaseline(input_dim=X_train.shape[2], ode_unfolds=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

EPOCHS = 30
best_val_loss = float("inf")
patience = 5
patience_counter = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            val_losses.append(criterion(pred, yb).item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "ltc_baseline_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

model.load_state_dict(torch.load("ltc_baseline_best.pt"))
model.eval()

all_preds, all_targets = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        pred = model(xb.to(device))
        all_preds.append(pred.numpy())
        all_targets.append(yb.numpy())

preds_scaled = np.concatenate(all_preds)
targets_scaled = np.concatenate(all_targets)

preds_log = preds_scaled * y_log_std + y_log_mean
targets_log = targets_scaled * y_log_std + y_log_mean
preds_ms = np.expm1(preds_log)
targets_ms = np.expm1(targets_log)

mae = np.mean(np.abs(preds_ms - targets_ms))
rmse = np.sqrt(np.mean((preds_ms - targets_ms) ** 2))
mape = np.mean(np.abs((preds_ms - targets_ms) / np.clip(targets_ms, 1e-3, None))) * 100
pearson_corr = np.corrcoef(preds_ms, targets_ms)[0, 1]

print(f"\n=== Test set results (real ms) ===")
print(f"MAE:  {mae:.2f} ms")
print(f"RMSE: {rmse:.2f} ms")
print(f"MAPE: {mape:.2f}%")
print(f"Pearson correlation: {pearson_corr:.4f}")

naive_mae = np.mean(np.abs(targets_ms - np.expm1(y_log_mean)))
print(f"\nNaive baseline MAE: {naive_mae:.2f} ms")
print(f"LTC model MAE:      {mae:.2f} ms  (compare: LSTM 19.49, GRU 18.22, XGBoost 21.23, CfC 17.84)")

np.save("ltc_test_preds_ms.npy", preds_ms)
np.save("ltc_test_targets_ms.npy", targets_ms)
print("\nSaved ltc_baseline_best.pt, ltc_test_preds_ms.npy, ltc_test_targets_ms.npy")
