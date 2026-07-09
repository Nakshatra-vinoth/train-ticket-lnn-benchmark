import json

WINDOW_SIZE = 50
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = remaining 0.15

with open("windows_index.jsonl") as f:
    windows = [json.loads(l) for l in f]

# Windows are already in chronological order (generated in order from build_windows.py)
n = len(windows)
print(f"Total valid windows: {n}")

train_end = int(n * TRAIN_FRAC)
val_end = train_end + int(n * VAL_FRAC)

# Purge: drop WINDOW_SIZE windows at each split boundary so no window in one split
# shares any record with a window in an adjacent split
purge = WINDOW_SIZE

train = windows[0 : train_end - purge]
val = windows[train_end + purge : val_end - purge]
test = windows[val_end + purge :]

print(f"\nTrain windows: {len(train)}  (indices 0 to {train_end - purge})")
print(f"Purge zone 1:  {purge} windows dropped")
print(f"Val windows:   {len(val)}  (indices {train_end + purge} to {val_end - purge})")
print(f"Purge zone 2:  {purge} windows dropped")
print(f"Test windows:  {len(test)}  (indices {val_end + purge} to {n})")

total_kept = len(train) + len(val) + len(test)
total_purged = n - total_kept
print(f"\nTotal windows kept: {total_kept} ({total_kept/n*100:.1f}%)")
print(f"Total windows purged: {total_purged} ({total_purged/n*100:.1f}%)")

# Sanity check: confirm no shared trace_ids across splits (would indicate a leakage bug)
def all_trace_ids(split):
    ids = set()
    for w in split:
        ids.update(w["context_trace_ids"])
        ids.add(w["target_trace_id"])
    return ids

train_ids = all_trace_ids(train)
val_ids = all_trace_ids(val)
test_ids = all_trace_ids(test)

print(f"\nOverlap check (should all be 0):")
print(f"  train ∩ val:  {len(train_ids & val_ids)}")
print(f"  train ∩ test: {len(train_ids & test_ids)}")
print(f"  val ∩ test:   {len(val_ids & test_ids)}")

with open("windows_train.jsonl", "w") as f:
    for w in train:
        f.write(json.dumps(w) + "\n")
with open("windows_val.jsonl", "w") as f:
    for w in val:
        f.write(json.dumps(w) + "\n")
with open("windows_test.jsonl", "w") as f:
    for w in test:
        f.write(json.dumps(w) + "\n")

print("\nSaved windows_train.jsonl, windows_val.jsonl, windows_test.jsonl")
