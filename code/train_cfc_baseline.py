import numpy as np
import torch
import torch.nn as nn
import time, json
import pandas as pd
from ncps.torch import CfC
from ncps.wirings import AutoNCP
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load frozen benchmark artifacts
X_train = np.load("X_train_scaled.npy")   # (N, 49, 27)
X_val   = np.load("X_val_scaled.npy")
X_test  = np.load("X_test_scaled.npy")
y_train = np.load("y_train.npy")          # raw ms, same target as LSTM/GRU
y_val   = np.load("y_val.npy")
y_test  = np.load("y_test.npy")

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t   = torch.tensor(X_val, dtype=torch.float32)
X_test_t  = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
y_val_t   = torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1)
y_test_t  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

INPUT_SIZE = X_train.shape[2]   # 27
UNITS = 32

wiring = AutoNCP(UNITS, 1)  # 1 output unit
model = CfC(INPUT_SIZE, wiring, batch_first=True).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 5

train_ds = torch.utils.data.TensorDataset(X_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
history = []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out, _ = model(xb)
        pred = out[:, -1, :]  # last timestep output
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_out, _ = model(X_val_t.to(device))
        val_pred = val_out[:, -1, :]
        val_loss = loss_fn(val_pred, y_val_t.to(device)).item()

    train_loss = np.mean(train_losses)
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        torch.save(model.state_dict(), "cfc_baseline_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

total_time = time.time() - start_time

# Reload best checkpoint
model.load_state_dict(torch.load("cfc_baseline_best.pt"))
model.eval()

with torch.no_grad():
    test_out, _ = model(X_test_t.to(device))
    test_pred = test_out[:, -1, :].cpu().numpy().flatten()

y_test_np = y_test

mae = mean_absolute_error(y_test_np, test_pred)
rmse = np.sqrt(mean_squared_error(y_test_np, test_pred))
mape = np.mean(np.abs((y_test_np - test_pred) / np.clip(y_test_np, 1e-3, None))) * 100
r2 = r2_score(y_test_np, test_pred)
pearson_r, _ = pearsonr(test_pred, y_test_np)
naive_mae = np.mean(np.abs(y_test_np - np.median(y_train)))

print(f"\nCfC Test Metrics:")
print(f"  MAE:     {mae:.2f} ms")
print(f"  RMSE:    {rmse:.2f} ms")
print(f"  MAPE:    {mape:.2f}%")
print(f"  R2:      {r2:.4f}")
print(f"  Pearson: {pearson_r:.4f}")
print(f"  Naive MAE: {naive_mae:.2f} ms")
print(f"  Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
print(f"  Total training time: {total_time:.2f}s")

# Save history CSV
pd.DataFrame(history).to_csv("cfc_training_history.csv", index=False)

# Save summary
with open("cfc_training_summary.md", "w") as f:
    f.write(f"# CfC Training Summary\n\n")
    f.write(f"- Best validation loss: {best_val_loss:.4f}\n")
    f.write(f"- Best validation epoch: {best_epoch}\n")
    f.write(f"- Early stopping epoch: {len(history)}\n")
    f.write(f"- Total training time: {total_time:.2f} seconds\n\n")
    f.write(f"## Test Metrics\n\n")
    f.write(f"- MAE: {mae:.2f} ms\n")
    f.write(f"- RMSE: {rmse:.2f} ms\n")
    f.write(f"- MAPE: {mape:.2f}%\n")
    f.write(f"- R2: {r2:.4f}\n")
    f.write(f"- Pearson: {pearson_r:.4f}\n")
    f.write(f"- Naive MAE: {naive_mae:.2f} ms\n")

np.save("cfc_test_preds_ms.npy", test_pred)
np.save("cfc_test_targets_ms.npy", y_test_np)

print("\nSaved: cfc_baseline_best.pt, cfc_training_history.csv, cfc_training_summary.md")
print("Saved: cfc_test_preds_ms.npy, cfc_test_targets_ms.npy")
