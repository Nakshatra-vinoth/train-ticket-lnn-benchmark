import json
import numpy as np
import xgboost as xgb

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

def summarize_window(X):
    """X shape: (N, 49, 27) -> summary stats per feature -> (N, 27*5)"""
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    last = X[:, -1, :]
    return np.concatenate([mean, std, mn, mx, last], axis=1)

X_train_flat = summarize_window(X_train)
X_val_flat = summarize_window(X_val)
X_test_flat = summarize_window(X_test)

print(f"Flattened feature dim: {X_train_flat.shape[1]} (should be 27*5=135)")

dtrain = xgb.DMatrix(X_train_flat, label=y_train)
dval = xgb.DMatrix(X_val_flat, label=y_val)
dtest = xgb.DMatrix(X_test_flat, label=y_test)

params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

evals_result = {}
model = xgb.train(
    params, dtrain,
    num_boost_round=500,
    evals=[(dtrain, "train"), (dval, "val")],
    early_stopping_rounds=20,
    evals_result=evals_result,
    verbose_eval=20,
)

print(f"\nBest iteration: {model.best_iteration}")

preds_scaled = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
targets_scaled = y_test

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
print(f"R²:   {r2:.4f}")
print(f"MAE:  {mae:.2f} ms")
print(f"RMSE: {rmse:.2f} ms")
print(f"MAPE: {mape:.2f}%")
print(f"Pearson correlation: {pearson_corr:.4f}")

naive_mae = np.mean(np.abs(targets_ms - np.expm1(y_log_mean)))
print(f"\nNaive baseline MAE: {naive_mae:.2f} ms")
print(f"XGBoost model MAE: {mae:.2f} ms")

model.save_model("xgboost_baseline.json")
np.save("xgboost_test_preds_ms.npy", preds_ms)
np.save("xgboost_test_targets_ms.npy", targets_ms)
print("\nSaved xgboost_baseline.json, xgboost_test_preds_ms.npy, xgboost_test_targets_ms.npy")
