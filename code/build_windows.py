import json
import numpy as np

WINDOW_SIZE = 50   # 49 context requests + 1 target request
GAP_THRESHOLD = 60.0  # seconds; any delta_t above this invalidates the window
STRIDE = 1  # slide by 1 request each time (max data, overlapping windows)

with open("dataset_with_deltat.jsonl") as f:
    records = [json.loads(l) for l in f]

print(f"Total records: {len(records)}")

n = len(records)
windows = []
skipped_gap = 0

# A window spans indices [i, i+WINDOW_SIZE-1] as context, target = record at i+WINDOW_SIZE
for i in range(0, n - WINDOW_SIZE, STRIDE):
    context = records[i : i + WINDOW_SIZE - 1]      # 49 records
    target = records[i + WINDOW_SIZE - 1]            # the 50th record (what we predict)

    # Check every delta_t within this span (context + target) for a gap violation
    span = records[i : i + WINDOW_SIZE]  # all 50 records in this window
    has_gap = any(r["delta_t"] > GAP_THRESHOLD for r in span[1:])  # skip first record's own delta_t (relative to record before window)

    if has_gap:
        skipped_gap += 1
        continue

    windows.append({
        "context_trace_ids": [r["trace_id"] for r in context],
        "target_trace_id": target["trace_id"],
        "window_start_idx": i,
    })

print(f"\nTotal candidate windows: {n - WINDOW_SIZE}")
print(f"Windows skipped due to gap > {GAP_THRESHOLD}s: {skipped_gap}")
print(f"Valid windows generated: {len(windows)}")
print(f"Skip rate: {skipped_gap / (n - WINDOW_SIZE) * 100:.2f}%")

# Save just the window index structure for now (not full feature vectors yet —
# that's the next step, once this shape is confirmed correct)
with open("windows_index.jsonl", "w") as f:
    for w in windows:
        f.write(json.dumps(w) + "\n")

print("\nSaved window index to windows_index.jsonl")
