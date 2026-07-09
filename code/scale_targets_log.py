import numpy as np
import json

y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
y_test = np.load("y_test.npy")

# Log-transform first (compresses heavy right tail), THEN standardize
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)
y_test_log = np.log1p(y_test)

y_log_mean = y_train_log.mean()
y_log_std = y_train_log.std() + 1e-8

print(f"Raw y (train)      - mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
print(f"log1p(y) (train)   - mean: {y_log_mean:.4f}, std: {y_log_std:.4f}")

y_train_scaled = (y_train_log - y_log_mean) / y_log_std
y_val_scaled = (y_val_log - y_log_mean) / y_log_std
y_test_scaled = (y_test_log - y_log_mean) / y_log_std

np.save("y_train_scaled.npy", y_train_scaled)
np.save("y_val_scaled.npy", y_val_scaled)
np.save("y_test_scaled.npy", y_test_scaled)

# Update scaler_params.json with log-scale params (keep feature params, replace y params)
with open("scaler_params.json") as f:
    scaler = json.load(f)

scaler["y_log_mean"] = float(y_log_mean)
scaler["y_log_std"] = float(y_log_std)
scaler["y_transform"] = "log1p_then_standardize"
# Remove old raw-scale params to avoid accidentally using them later
scaler.pop("y_mean", None)
scaler.pop("y_std", None)

with open("scaler_params.json", "w") as f:
    json.dump(scaler, f, indent=2)

print("\nUpdated y_{train,val,test}_scaled.npy and scaler_params.json (now log1p + standardize)")
