import json
import numpy as np

WINDOW_SIZE = 50

ROOT_SERVICES = sorted(['ts-travel-service', 'ts-travel2-service', 'ts-basic-service'])
ALL_SERVICES = sorted([
    'ts-basic-service', 'ts-station-service', 'ts-route-service', 'ts-train-service',
    'ts-travel2-service', 'ts-config-service', 'ts-seat-service', 'ts-price-service',
    'ts-order-other-service', 'ts-travel-service', 'ts-order-service'
])

# Dropped: rolling_error_rate (always 0, no information)
# Dropped: critical_path_latency_ms (100% identical to end_to_end_latency_ms, pure redundancy)
NUMERIC_FIELDS = [
    'delta_t', 'end_to_end_latency_ms', 'num_spans', 'trace_depth', 'num_unique_services',
    'rolling_request_rate',
    'rolling_latency_mean_ms', 'rolling_latency_p95_ms', 'rolling_latency_p99_ms'
]

def aggregate_system_metrics(record):
    services = record["services_involved"]
    sm = record["system_metrics"]
    cpu_vals, mem_vals, rx_vals, tx_vals = [], [], [], []
    for svc in services:
        if svc in sm:
            cpu_vals.append(sm[svc]["cpu_percent"])
            mem_vals.append(sm[svc]["mem_percent"])
            rx_vals.append(sm[svc]["net_rx_bytes"])
            tx_vals.append(sm[svc]["net_tx_bytes"])
    if not cpu_vals:
        return [0.0, 0.0, 0.0, 0.0]
    return [np.mean(cpu_vals), np.mean(mem_vals), np.mean(rx_vals), np.mean(tx_vals)]

def encode_root_service(record):
    vec = [0.0] * len(ROOT_SERVICES)
    if record["root_service"] in ROOT_SERVICES:
        vec[ROOT_SERVICES.index(record["root_service"])] = 1.0
    return vec

def encode_services_involved(record):
    vec = [0.0] * len(ALL_SERVICES)
    for svc in record["services_involved"]:
        if svc in ALL_SERVICES:
            vec[ALL_SERVICES.index(svc)] = 1.0
    return vec

def build_feature_vector(record):
    numeric = [float(record[f]) for f in NUMERIC_FIELDS]
    sysm = aggregate_system_metrics(record)
    root_oh = encode_root_service(record)
    svc_mh = encode_services_involved(record)
    return numeric + sysm + root_oh + svc_mh

FEATURE_DIM = len(NUMERIC_FIELDS) + 4 + len(ROOT_SERVICES) + len(ALL_SERVICES)
print(f"Feature dimension per timestep: {FEATURE_DIM} (was 29, now {FEATURE_DIM})")

print("Loading full dataset...")
with open("dataset_with_deltat.jsonl") as f:
    all_records = [json.loads(l) for l in f]
by_trace_id = {r["trace_id"]: r for r in all_records}
print(f"Indexed {len(by_trace_id)} records")

def process_split(split_name):
    with open(f"windows_{split_name}.jsonl") as f:
        window_defs = [json.loads(l) for l in f]

    n = len(window_defs)
    X = np.zeros((n, WINDOW_SIZE - 1, FEATURE_DIM), dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    for idx, w in enumerate(window_defs):
        for t, tid in enumerate(w["context_trace_ids"]):
            record = by_trace_id[tid]
            X[idx, t, :] = build_feature_vector(record)
        target_record = by_trace_id[w["target_trace_id"]]
        y[idx] = target_record["end_to_end_latency_ms"]

    np.save(f"X_{split_name}.npy", X)
    np.save(f"y_{split_name}.npy", y)
    print(f"{split_name}: X shape={X.shape}, y shape={y.shape}")

for split in ["train", "val", "test"]:
    process_split(split)

print("\nDone. Saved X_{train,val,test}.npy and y_{train,val,test}.npy")
