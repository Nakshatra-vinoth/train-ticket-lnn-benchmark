import json
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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

model = GRUBaseline(input_dim=X_train.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

EPOCHS = 30
best_val_loss = float("inf")
best_val_epoch = None
patience = 5
patience_counter = 0
early_stopping_epoch = None
history = []
training_start = time.time()

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
    print(f"Epoch {epoch:2d}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
    history.append({
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
    })

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), "gru_baseline_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            early_stopping_epoch = epoch
            break

training_time_seconds = time.time() - training_start

print(f"\nBest validation loss: {best_val_loss:.4f}")
print(f"Best validation epoch: {best_val_epoch}")
print(f"Training time: {training_time_seconds:.2f} seconds")

with open("gru_training_history.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
    writer.writeheader()
    writer.writerows(history)

model.load_state_dict(torch.load("gru_baseline_best.pt"))
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
ss_res = np.sum((targets_ms - preds_ms) ** 2)
ss_tot = np.sum((targets_ms - np.mean(targets_ms)) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"\n=== Test set results (real ms) ===")
print(f"MAE:  {mae:.2f} ms")
print(f"RMSE: {rmse:.2f} ms")
print(f"MAPE: {mape:.2f}%")
print(f"Pearson correlation: {pearson_corr:.4f}")
print(f"R2:   {r2:.4f}")

naive_mae = np.mean(np.abs(targets_ms - np.expm1(y_log_mean)))
print(f"\nNaive baseline MAE: {naive_mae:.2f} ms")
print(f"GRU model MAE:      {mae:.2f} ms")

np.save("gru_test_preds_ms.npy", preds_ms)
np.save("gru_test_targets_ms.npy", targets_ms)

with open("gru_training_summary.md", "w") as f:
    f.write("# GRU Training Summary\n\n")
    f.write("Frozen benchmark: yes\n\n")
    f.write(f"- Best validation loss: {best_val_loss:.6f}\n")
    f.write(f"- Best validation epoch: {best_val_epoch}\n")
    if early_stopping_epoch is None:
        f.write("- Early stopping epoch: not triggered\n")
    else:
        f.write(f"- Early stopping epoch: {early_stopping_epoch}\n")
    f.write(f"- Total training time seconds: {training_time_seconds:.2f}\n")
    f.write("\n## Test Metrics\n\n")
    f.write(f"- MAE: {mae:.2f} ms\n")
    f.write(f"- RMSE: {rmse:.2f} ms\n")
    f.write(f"- MAPE: {mape:.2f}%\n")
    f.write(f"- R2: {r2:.4f}\n")
    f.write(f"- Pearson correlation: {pearson_corr:.4f}\n")
    f.write(f"- Naive MAE: {naive_mae:.2f} ms\n")

print("\nSaved gru_baseline_best.pt, gru_test_preds_ms.npy, gru_test_targets_ms.npy")
print("Saved gru_training_history.csv, gru_training_summary.md")
