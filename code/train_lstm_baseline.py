import json
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

# ----------------------------
# Load data
# ----------------------------
X_train = np.load("X_train_scaled.npy")
X_val = np.load("X_val_scaled.npy")
X_test = np.load("X_test_scaled.npy")

y_train = np.load("y_train_scaled.npy")
y_val = np.load("y_val_scaled.npy")
y_test = np.load("y_test_scaled.npy")

with open("scaler_params.json") as f:
    scaler = json.load(f)

y_log_mean = scaler["y_log_mean"]
y_log_std = scaler["y_log_std"]

print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")

device = torch.device("cpu")
print(f"Device: {device}")

# ----------------------------
# Data loaders
# ----------------------------
def make_loader(X, y, batch_size=64, shuffle=False):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
    )

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader = make_loader(X_val, y_val)
test_loader = make_loader(X_test, y_test)

# ----------------------------
# Model
# ----------------------------
class LSTMBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1]
        return self.head(last_hidden).squeeze(-1)

model = LSTMBaseline(input_dim=X_train.shape[2]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# ----------------------------
# Training
# ----------------------------
EPOCHS = 30
PATIENCE = 5

best_val_loss = float("inf")
best_val_epoch = None
patience_counter = 0
early_stopping_epoch = None
history = []
training_start = time.time()

for epoch in range(1, EPOCHS + 1):

    model.train()
    train_losses = []

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

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

            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)

            val_losses.append(
                criterion(pred, yb).item()
            )

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    print(
        f"Epoch {epoch:02d} | "
        f"train={train_loss:.4f} | "
        f"val={val_loss:.4f}"
    )
    history.append({
        "epoch": epoch,
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
    })

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_val_epoch = epoch
        patience_counter = 0

        torch.save(model.state_dict(), "lstm_baseline_best.pt")

    else:

        patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            early_stopping_epoch = epoch
            break

training_time_seconds = time.time() - training_start

print(f"\nBest validation loss: {best_val_loss:.4f}")
print(f"Best validation epoch: {best_val_epoch}")
print(f"Training time: {training_time_seconds:.2f} seconds")

with open("lstm_training_history.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
    writer.writeheader()
    writer.writerows(history)

# ----------------------------
# Evaluation
# ----------------------------
model.load_state_dict(torch.load("lstm_baseline_best.pt"))
model.eval()

preds = []
targets = []

with torch.no_grad():

    for xb, yb in test_loader:

        pred = model(xb.to(device))

        preds.append(pred.cpu().numpy())
        targets.append(yb.numpy())

preds_scaled = np.concatenate(preds)
targets_scaled = np.concatenate(targets)

# Undo standardization
preds_log = preds_scaled * y_log_std + y_log_mean
targets_log = targets_scaled * y_log_std + y_log_mean

# Undo log1p
preds_ms = np.expm1(preds_log)
targets_ms = np.expm1(targets_log)

# ----------------------------
# Metrics
# ----------------------------
mae = np.mean(np.abs(preds_ms - targets_ms))

rmse = np.sqrt(
    np.mean((preds_ms - targets_ms) ** 2)
)

mape = (
    np.mean(
        np.abs(
            (preds_ms - targets_ms)
            / np.clip(targets_ms, 1e-3, None)
        )
    )
    * 100
)

pearson = np.corrcoef(preds_ms, targets_ms)[0, 1]

ss_res = np.sum((targets_ms - preds_ms) ** 2)
ss_tot = np.sum((targets_ms - np.mean(targets_ms)) ** 2)
r2 = 1 - ss_res / ss_tot

naive_mae = np.mean(
    np.abs(
        targets_ms - np.expm1(y_log_mean)
    )
)

print("\n========== TEST RESULTS ==========")
print(f"MAE:                 {mae:.2f} ms")
print(f"RMSE:                {rmse:.2f} ms")
print(f"MAPE:                {mape:.2f}%")
print(f"Pearson correlation: {pearson:.4f}")
print(f"R²:                  {r2:.4f}")

print("\n========== BASELINE ==========")
print(f"Naive MAE:           {naive_mae:.2f} ms")

np.save("lstm_test_preds_ms.npy", preds_ms)
np.save("lstm_test_targets_ms.npy", targets_ms)

print("\nSaved:")
print("  lstm_baseline_best.pt")
print("  lstm_test_preds_ms.npy")
print("  lstm_test_targets_ms.npy")
print("  lstm_training_history.csv")

with open("lstm_training_summary.md", "w") as f:
    f.write("# LSTM Training Summary\n\n")
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
    f.write(f"- Pearson correlation: {pearson:.4f}\n")
    f.write(f"- Naive MAE: {naive_mae:.2f} ms\n")

print("  lstm_training_summary.md")
