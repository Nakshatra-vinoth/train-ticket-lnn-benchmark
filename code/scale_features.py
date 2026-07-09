import numpy as np
import json

CONTINUOUS_IDX = list(range(0, 13))  # 9 numeric fields + 4 system-metric aggregates
BINARY_IDX = list(range(13, 27))     # 3 root_service one-hot + 11 services_involved multi-hot

X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
y_test = np.load("y_test.npy")

print(f"X_train shape: {X_train.shape}")

train_continuous = X_train[:, :, CONTINUOUS_IDX]
feat_mean = train_continuous.mean(axis=(0, 1))
feat_std = train_continuous.std(axis=(0, 1)) + 1e-8

print(f"\nContinuous feature means (train): {feat_mean}")
print(f"Continuous feature stds (train):  {feat_std}")

def scale_X(X):
    X_scaled = X.copy()
    X_scaled[:, :, CONTINUOUS_IDX] = (X[:, :, CONTINUOUS_IDX] - feat_mean) / feat_std
    return X_scaled

X_train_scaled = scale_X(X_train)
X_val_scaled = scale_X(X_val)
X_test_scaled = scale_X(X_test)

y_mean = y_train.mean()
y_std = y_train.std() + 1e-8
print(f"\ny (latency) mean (train): {y_mean:.4f}, std: {y_std:.4f}")

y_train_scaled = (y_train - y_mean) / y_std
y_val_scaled = (y_val - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

print(f"\nPost-scaling train continuous mean (should be ~0): {X_train_scaled[:,:,CONTINUOUS_IDX].mean(axis=(0,1))[:3]}...")
print(f"Post-scaling train continuous std (should be ~1):  {X_train_scaled[:,:,CONTINUOUS_IDX].std(axis=(0,1))[:3]}...")

np.save("X_train_scaled.npy", X_train_scaled)
np.save("X_val_scaled.npy", X_val_scaled)
np.save("X_test_scaled.npy", X_test_scaled)
np.save("y_train_scaled.npy", y_train_scaled)
np.save("y_val_scaled.npy", y_val_scaled)
np.save("y_test_scaled.npy", y_test_scaled)

scaler_params = {
    "feature_mean": feat_mean.tolist(),
    "feature_std": feat_std.tolist(),
    "continuous_idx": CONTINUOUS_IDX,
    "y_mean": float(y_mean),
    "y_std": float(y_std),
}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f, indent=2)

print("\nSaved scaled arrays and scaler_params.json")
