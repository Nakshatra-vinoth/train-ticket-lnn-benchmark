import json
from datetime import datetime, timezone

INPUT_FILE = "dataset_with_deltat.jsonl"
OUTPUT_FILE = "dataset_with_deltat_trimmed.jsonl"

# Load dataset
with open(INPUT_FILE) as f:
    records = [json.loads(line) for line in f]

# Sort by timestamp
records.sort(key=lambda r: r["request_timestamp"])

# Find the last ts-travel-service record
last_index = None
for i, record in enumerate(records):
    if record["root_service"] == "ts-travel-service":
        last_index = i

if last_index is None:
    raise RuntimeError("No ts-travel-service records found!")

# Keep everything up to and including that record
trimmed = records[:last_index + 1]

# Save trimmed dataset
with open(OUTPUT_FILE, "w") as f:
    for record in trimmed:
        f.write(json.dumps(record) + "\n")

# Print summary
first_time = datetime.fromtimestamp(
    trimmed[0]["request_timestamp"], tz=timezone.utc
)
last_time = datetime.fromtimestamp(
    trimmed[-1]["request_timestamp"], tz=timezone.utc
)

print(f"Original records : {len(records)}")
print(f"Trimmed records  : {len(trimmed)}")
print(f"Removed records  : {len(records) - len(trimmed)}")
print(f"Time range kept  : {first_time.isoformat()} -> {last_time.isoformat()}")

# Count services
counts = {}
for r in trimmed:
    svc = r["root_service"]
    counts[svc] = counts.get(svc, 0) + 1

print("\nService counts:")
for svc, cnt in sorted(counts.items()):
    print(f"{svc:25s} {cnt}")
