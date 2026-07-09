import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SOURCE_DATASET = Path("backup_before_trim/dataset_with_deltat.jsonl")
FROZEN_DATASET = Path("dataset_with_deltat.jsonl")
SUMMARY_PATH = Path("benchmark_split_summary.json")

WINDOW_SIZE = 50
GAP_THRESHOLD_SECONDS = 60.0
PURGE_WINDOWS = 50

MIN_TRAIN_FRAC = 0.50
MIN_VAL_FRAC = 0.10
MIN_TEST_FRAC = 0.10
TARGET_SPLIT_FRACS = np.array([0.70, 0.15, 0.15], dtype=float)

LATENCY_QUANTILES = [50, 75, 90, 95, 99]
REQUEST_RATE_QUANTILES = [50, 75, 90, 95, 99]


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def load_records(path):
    with path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]
    records.sort(key=lambda r: r["request_timestamp"])
    return records


def required_root_services(records):
    return sorted({r["root_service"] for r in records})


def select_contiguous_interval(records, roots):
    """Drop suffix after the first permanent disappearance of any required root."""
    last_idx_by_root = {}
    for root in roots:
        indices = [i for i, r in enumerate(records) if r["root_service"] == root]
        if not indices:
            raise RuntimeError(f"Required root service has no records: {root}")
        last_idx_by_root[root] = indices[-1]

    end_idx = min(last_idx_by_root.values())
    disappeared_roots = [root for root, idx in last_idx_by_root.items() if idx == end_idx]
    return records[: end_idx + 1], end_idx, last_idx_by_root, disappeared_roots


def build_windows(records):
    windows = []
    skipped_gap = 0

    for start in range(0, len(records) - WINDOW_SIZE):
        span = records[start : start + WINDOW_SIZE]

        # The first record's delta_t is a feature, so it must be checked too.
        if any(float(r["delta_t"]) > GAP_THRESHOLD_SECONDS for r in span):
            skipped_gap += 1
            continue

        context = span[: WINDOW_SIZE - 1]
        target = span[WINDOW_SIZE - 1]
        windows.append(
            {
                "context_trace_ids": [r["trace_id"] for r in context],
                "target_trace_id": target["trace_id"],
                "window_start_idx": start,
            }
        )

    return windows, skipped_gap


def window_rows(windows, by_trace):
    rows = []
    for window in windows:
        target = by_trace[window["target_trace_id"]]
        rows.append(
            {
                "start": window["window_start_idx"],
                "ts": float(target["request_timestamp"]),
                "root": target["root_service"],
                "latency": float(target["end_to_end_latency_ms"]),
                "request_rate": float(target["rolling_request_rate"]),
            }
        )
    return rows


def split_ranges(n, train_end, val_end):
    return (
        range(0, train_end - PURGE_WINDOWS),
        range(train_end + PURGE_WINDOWS, val_end - PURGE_WINDOWS),
        range(val_end + PURGE_WINDOWS, n),
    )


def split_windows(windows, train_end, val_end):
    train_idx, val_idx, test_idx = split_ranges(len(windows), train_end, val_end)
    return (
        [windows[i] for i in train_idx],
        [windows[i] for i in val_idx],
        [windows[i] for i in test_idx],
    )


def stats_for_indices(rows, roots, indices):
    indices = list(indices)
    latencies = np.array([rows[i]["latency"] for i in indices], dtype=float)
    log_latencies = np.log1p(latencies)
    request_rates = np.array([rows[i]["request_rate"] for i in indices], dtype=float)
    counts = Counter(rows[i]["root"] for i in indices)
    n = len(indices)

    return {
        "n": n,
        "start_window_index": int(indices[0]),
        "end_window_index": int(indices[-1]),
        "start_timestamp": iso(rows[indices[0]]["ts"]),
        "end_timestamp": iso(rows[indices[-1]]["ts"]),
        "service_counts": {root: int(counts[root]) for root in roots},
        "service_percent": {root: float(counts[root] / n) for root in roots},
        "latency_ms": {
            "mean": float(latencies.mean()),
            "std": float(latencies.std()),
            "min": float(latencies.min()),
            "p50": float(np.percentile(latencies, 50)),
            "p75": float(np.percentile(latencies, 75)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "max": float(latencies.max()),
        },
        "log_latency": {
            "mean": float(log_latencies.mean()),
            "std": float(log_latencies.std()),
        },
        "request_rate": {
            "mean": float(request_rates.mean()),
            "std": float(request_rates.std()),
            "min": float(request_rates.min()),
            "p50": float(np.percentile(request_rates, 50)),
            "p75": float(np.percentile(request_rates, 75)),
            "p90": float(np.percentile(request_rates, 90)),
            "p95": float(np.percentile(request_rates, 95)),
            "p99": float(np.percentile(request_rates, 99)),
            "max": float(request_rates.max()),
        },
    }


def score_candidate(rows, roots, full_stats, train_end, val_end):
    n = len(rows)
    train_idx, val_idx, test_idx = split_ranges(n, train_end, val_end)
    lengths = np.array([len(list(train_idx)), len(list(val_idx)), len(list(test_idx))])

    if lengths.min() < 1000:
        return None

    fractions = lengths / lengths.sum()
    if np.any(fractions < np.array([MIN_TRAIN_FRAC, MIN_VAL_FRAC, MIN_TEST_FRAC])):
        return None

    train_idx, val_idx, test_idx = split_ranges(n, train_end, val_end)
    split_stats = [
        stats_for_indices(rows, roots, train_idx),
        stats_for_indices(rows, roots, val_idx),
        stats_for_indices(rows, roots, test_idx),
    ]

    full_pct = np.array([full_stats["service_percent"][root] for root in roots])
    service_shift = max(
        float(
            np.max(
                np.abs(
                    np.array([split["service_percent"][root] for root in roots])
                    - full_pct
                )
            )
        )
        for split in split_stats
    )

    latency_keys = ["p50", "p75", "p90", "p95", "p99"]
    rate_keys = ["p50", "p75", "p90", "p95", "p99"]
    full_latency_q = np.array([full_stats["latency_ms"][key] for key in latency_keys])
    full_rate_q = np.array([full_stats["request_rate"][key] for key in rate_keys])

    latency_quantile_shift = max(
        float(
            np.max(
                np.abs(
                    np.array([split["latency_ms"][key] for key in latency_keys])
                    - full_latency_q
                )
                / np.maximum(np.abs(full_latency_q), 1e-6)
            )
        )
        for split in split_stats
    )

    request_rate_quantile_shift = max(
        float(
            np.max(
                np.abs(
                    np.array([split["request_rate"][key] for key in rate_keys])
                    - full_rate_q
                )
                / np.maximum(np.abs(full_rate_q), 1e-6)
            )
        )
        for split in split_stats
    )

    log_latency_mean_shift = max(
        abs(split["log_latency"]["mean"] - full_stats["log_latency"]["mean"])
        for split in split_stats
    )
    log_latency_std_shift = max(
        abs(split["log_latency"]["std"] - full_stats["log_latency"]["std"])
        for split in split_stats
    )
    request_rate_mean_shift = max(
        abs(split["request_rate"]["mean"] - full_stats["request_rate"]["mean"])
        / (abs(full_stats["request_rate"]["mean"]) + 1e-6)
        for split in split_stats
    )

    continuity_loss = (n - lengths.sum()) / n
    split_size_shift = float(np.max(np.abs(fractions - TARGET_SPLIT_FRACS)))

    total = (
        3.0 * service_shift
        + 1.5 * latency_quantile_shift
        + 1.0 * log_latency_mean_shift
        + 1.0 * log_latency_std_shift
        + 1.0 * request_rate_quantile_shift
        + 0.5 * request_rate_mean_shift
        + 0.5 * continuity_loss
        + 0.25 * split_size_shift
    )

    return {
        "score": float(total),
        "train_end": int(train_end),
        "val_end": int(val_end),
        "kept_windows": int(lengths.sum()),
        "split_fractions": [float(x) for x in fractions],
        "components": {
            "service_shift": float(service_shift),
            "latency_quantile_shift": float(latency_quantile_shift),
            "log_latency_mean_shift": float(log_latency_mean_shift),
            "log_latency_std_shift": float(log_latency_std_shift),
            "request_rate_quantile_shift": float(request_rate_quantile_shift),
            "request_rate_mean_shift": float(request_rate_mean_shift),
            "continuity_loss": float(continuity_loss),
            "split_size_shift": float(split_size_shift),
        },
        "split_stats": {
            "train": split_stats[0],
            "val": split_stats[1],
            "test": split_stats[2],
        },
    }


def search_splits(rows, roots):
    n = len(rows)
    full_stats = stats_for_indices(rows, roots, range(n))
    candidates = []

    for train_frac in np.arange(0.50, 0.781, 0.0025):
        train_end = int(n * train_frac)
        for val_frac in np.arange(0.10, 0.251, 0.0025):
            val_end = train_end + int(n * val_frac)
            if val_end >= n - PURGE_WINDOWS - 1000:
                continue
            candidate = score_candidate(rows, roots, full_stats, train_end, val_end)
            if candidate is not None:
                candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No valid split candidates found")

    candidates.sort(key=lambda c: (c["score"], c["train_end"], c["val_end"]))
    return candidates, full_stats


def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    records = load_records(SOURCE_DATASET)
    roots = required_root_services(records)
    interval, end_idx, last_idx_by_root, disappeared_roots = select_contiguous_interval(
        records, roots
    )

    by_trace = {r["trace_id"]: r for r in interval}
    windows, skipped_gap = build_windows(interval)
    rows = window_rows(windows, by_trace)
    candidates, full_window_stats = search_splits(rows, roots)
    selected = candidates[0]
    train, val, test = split_windows(
        windows, selected["train_end"], selected["val_end"]
    )

    write_jsonl(FROZEN_DATASET, interval)
    write_jsonl(Path("windows_index.jsonl"), windows)
    write_jsonl(Path("windows_train.jsonl"), train)
    write_jsonl(Path("windows_val.jsonl"), val)
    write_jsonl(Path("windows_test.jsonl"), test)

    summary = {
        "source_dataset": str(SOURCE_DATASET),
        "frozen_dataset": str(FROZEN_DATASET),
        "source_records": len(records),
        "source_start_timestamp": iso(records[0]["request_timestamp"]),
        "source_end_timestamp": iso(records[-1]["request_timestamp"]),
        "required_root_services": roots,
        "service_continuity_rule": (
            "Select the prefix ending at the earliest last occurrence among all "
            "required root services, thereby excluding any suffix in which one or "
            "more required root services have permanently disappeared."
        ),
        "last_index_by_root_service": last_idx_by_root,
        "interval_end_index": end_idx,
        "interval_end_root_services": disappeared_roots,
        "interval_records": len(interval),
        "interval_start_timestamp": iso(interval[0]["request_timestamp"]),
        "interval_end_timestamp": iso(interval[-1]["request_timestamp"]),
        "removed_suffix_records": len(records) - len(interval),
        "window_size": WINDOW_SIZE,
        "context_steps": WINDOW_SIZE - 1,
        "gap_threshold_seconds": GAP_THRESHOLD_SECONDS,
        "valid_windows": len(windows),
        "windows_skipped_for_gap": skipped_gap,
        "purge_windows": PURGE_WINDOWS,
        "objective": {
            "score": (
                "3.0*service_shift + 1.5*latency_quantile_shift + "
                "1.0*log_latency_mean_shift + 1.0*log_latency_std_shift + "
                "1.0*request_rate_quantile_shift + 0.5*request_rate_mean_shift + "
                "0.5*continuity_loss + 0.25*split_size_shift"
            ),
            "latency_quantiles": LATENCY_QUANTILES,
            "request_rate_quantiles": REQUEST_RATE_QUANTILES,
            "minimum_split_fractions": {
                "train": MIN_TRAIN_FRAC,
                "val": MIN_VAL_FRAC,
                "test": MIN_TEST_FRAC,
            },
            "target_split_fractions_for_tie_regularization": TARGET_SPLIT_FRACS.tolist(),
        },
        "full_window_stats": full_window_stats,
        "selected": selected,
        "top_candidates": candidates[:10],
        "frozen_before_final_model_training": True,
        "model_metrics_used_for_selection": False,
    }

    with SUMMARY_PATH.open("w") as f:
        json.dump(summary, f, indent=2)

    print("Benchmark construction complete")
    print(f"Source records: {len(records)}")
    print(f"Frozen interval records: {len(interval)}")
    print(f"Valid windows: {len(windows)}")
    print(f"Skipped windows due to gap: {skipped_gap}")
    print(
        f"Selected train_end={selected['train_end']} val_end={selected['val_end']} "
        f"score={selected['score']:.6f}"
    )
    print(f"Train/val/test windows: {len(train)} / {len(val)} / {len(test)}")
    print(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
