import json
import numpy as np
from datetime import datetime, timezone

with open("dataset.jsonl") as f:
    records = [json.loads(l) for l in f]

# Drop records with empty system_metrics (the 585 from the incident window)
before = len(records)
records = [r for r in records if r.get("system_metrics")]
print(f"Dropped {before - len(records)} records with empty system_metrics")

# Sort globally by timestamp (single pooled stream)
records.sort(key=lambda r: r["request_timestamp"])

# Compute delta_t for every record (first record gets delta_t = 0)
records[0]["delta_t"] = 0.0
for i in range(1, len(records)):
    records[i]["delta_t"] = records[i]["request_timestamp"] - records[i-1]["request_timestamp"]

delta_ts = np.array([r["delta_t"] for r in records[1:]])  # exclude the artificial first 0.0

print(f"\nTotal records: {len(records)}")
print(f"Delta_t stats (seconds):")
print(f"  min:    {delta_ts.min():.4f}")
print(f"  mean:   {delta_ts.mean():.4f}")
print(f"  median: {np.median(delta_ts):.4f}")
print(f"  p95:    {np.percentile(delta_ts, 95):.4f}")
print(f"  p99:    {np.percentile(delta_ts, 99):.4f}")
print(f"  max:    {delta_ts.max():.4f}")

large_gaps = delta_ts[delta_ts > 30]
print(f"\nDelta_t values > 30s: {len(large_gaps)}")
print(f"Delta_t values > 5s:  {len(delta_ts[delta_ts > 5])}")

with open("dataset_with_deltat.jsonl", "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

print("\nSaved to dataset_with_deltat.jsonl")