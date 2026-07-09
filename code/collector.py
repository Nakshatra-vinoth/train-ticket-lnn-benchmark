#!/usr/bin/env python3
"""
Incremental dataset collector for the Train Ticket LNN latency benchmark.
See module docstring in chat for full design rationale.
Run with: python3 collector.py --interval 30 --duration-hours 12 --output dataset.jsonl
"""
import requests
import argparse
import json
import time
from collections import deque
from datetime import datetime, timezone

from docker_stats_collector import get_container_stats

JAEGER_API = "http://localhost:16686/api"
PROMETHEUS_API = "http://localhost:9090/api/v1"

TRACED_SERVICES = [
    "ts-travel-service",
    "ts-travel2-service",
    "ts-basic-service",
]

ROLLING_WINDOW_SECONDS = 60


def is_synthetic_noise(num_spans, trace_depth, http_status, root_span_name):
    if http_status == 403 and num_spans <= 2:
        return True
    if root_span_name in ("HEAD", "GET") and trace_depth == 0 and num_spans <= 2:
        return True
    return False


def build_span_tree(trace):
    spans_by_id = {s["spanID"]: s for s in trace["spans"]}
    children = {}
    roots = []
    for s in trace["spans"]:
        parent_ids = [r["spanID"] for r in s.get("references", []) if r["refType"] == "CHILD_OF"]
        if parent_ids:
            children.setdefault(parent_ids[0], []).append(s["spanID"])
        else:
            roots.append(s["spanID"])
    return spans_by_id, children, roots


def compute_depth(span_id, children, depth=0):
    if span_id not in children or not children[span_id]:
        return depth
    return max(compute_depth(c, children, depth + 1) for c in children[span_id])


def compute_critical_path(span_id, spans_by_id, children):
    span = spans_by_id[span_id]
    own = span["duration"]
    if span_id not in children or not children[span_id]:
        return own
    return max(own, max(compute_critical_path(c, spans_by_id, children) for c in children[span_id]))


def get_service_name(span, trace):
    return trace["processes"][span["processID"]]["serviceName"]


def query_prometheus_snapshot(timestamp_unix):
    """
    Collect per-container CPU, memory and network metrics using
    `docker stats --no-stream` (fast, concurrent sampling — see
    docker_stats_collector.py). Keeps the same function name/signature
    so the rest of the collector does not change.
    """
    try:
        return get_container_stats()
    except Exception as e:
        print(f"[warn] docker stats snapshot failed: {e}")
        return {}


class RollingStats:
    def __init__(self, window_seconds=ROLLING_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self.events = deque()

    def add_and_compute(self, timestamp_sec, latency_ms, is_error):
        self.events.append((timestamp_sec, latency_ms, is_error))
        cutoff = timestamp_sec - self.window_seconds
        while self.events and self.events[0][0] < cutoff:
            self.events.popleft()

        n = len(self.events)
        if n == 0:
            return {
                "rolling_request_rate": 0.0,
                "rolling_error_rate": 0.0,
                "rolling_latency_mean_ms": 0.0,
                "rolling_latency_p95_ms": 0.0,
                "rolling_latency_p99_ms": 0.0,
            }

        latencies = sorted(e[1] for e in self.events)
        error_count = sum(1 for e in self.events if e[2])

        def percentile(sorted_vals, p):
            if not sorted_vals:
                return 0.0
            idx = min(int(len(sorted_vals) * p), len(sorted_vals) - 1)
            return sorted_vals[idx]

        return {
            "rolling_request_rate": n / self.window_seconds,
            "rolling_error_rate": error_count / n,
            "rolling_latency_mean_ms": sum(latencies) / n,
            "rolling_latency_p95_ms": percentile(latencies, 0.95),
            "rolling_latency_p99_ms": percentile(latencies, 0.99),
        }


def extract_record(trace, rolling_stats, system_snapshot):
    spans_by_id, children, roots = build_span_tree(trace)
    if len(roots) != 1:
        return None

    root_id = roots[0]
    root_span = spans_by_id[root_id]
    all_services = {get_service_name(s, trace) for s in trace["spans"]}

    error_status = False
    http_status = None
    for s in trace["spans"]:
        for tag in s.get("tags", []):
            if tag["key"] == "error" and tag.get("value") is True:
                error_status = True
            if tag["key"] in ("http.status_code", "http.status"):
                http_status = tag["value"]

    trace_depth = compute_depth(root_id, children)
    num_spans = len(trace["spans"])
    root_span_name = root_span["operationName"]

    if is_synthetic_noise(num_spans, trace_depth, http_status, root_span_name):
        return None

    critical_path_us = compute_critical_path(root_id, spans_by_id, children)
    start_time_us = root_span["startTime"]
    start_time_sec = start_time_us / 1_000_000.0
    latency_ms = root_span["duration"] / 1000.0

    rolling = rolling_stats.add_and_compute(start_time_sec, latency_ms, error_status)

    record = {
        "trace_id": trace["traceID"],
        "request_timestamp": start_time_sec,
        "request_timestamp_iso": datetime.fromtimestamp(start_time_sec, tz=timezone.utc).isoformat(),
        "root_span_name": root_span_name,
        "root_service": get_service_name(root_span, trace),
        "end_to_end_latency_ms": latency_ms,
        "num_spans": num_spans,
        "trace_depth": trace_depth,
        "num_unique_services": len(all_services),
        "services_involved": sorted(all_services),
        "critical_path_latency_ms": critical_path_us / 1000.0,
        "error_status": error_status,
        "http_status_code": http_status,
    }
    record.update(rolling)
    record["system_metrics"] = system_snapshot
    return record


def load_checkpoints(checkpoint_path):
    try:
        with open(checkpoint_path) as f:
            return json.load(f)
    except FileNotFoundError:
        now_us = int(time.time() * 1_000_000)
        return {svc: now_us for svc in TRACED_SERVICES}


def save_checkpoints(checkpoint_path, checkpoints):
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoints, f)


def fetch_new_traces(service, since_unix_us, limit=100):
    try:
        now_us = int(time.time() * 1_000_000)

        resp = requests.get(
            f"{JAEGER_API}/traces",
            params={
                "service": service,
                "limit": limit,
                "start": since_unix_us,
                "end": now_us,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["data"]
    except Exception as e:
        print(f"  [warn] fetch failed for {service}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--duration-hours", type=float, default=12.0)
    parser.add_argument("--output", type=str, default="dataset.jsonl")
    parser.add_argument("--checkpoint", type=str, default="collector_checkpoint.json")
    args = parser.parse_args()

    checkpoints = load_checkpoints(args.checkpoint)
    rolling_stats = RollingStats()
    seen_trace_ids = set()

    end_time = time.time() + args.duration_hours * 3600
    total_written = 0
    poll_count = 0

    print(f"[collector] starting. interval={args.interval}s duration={args.duration_hours}h output={args.output}")

    with open(args.output, "a") as outfile:
        while time.time() < end_time:
            poll_count += 1
            poll_start = time.time()
            new_records = []
            system_snapshot = query_prometheus_snapshot(time.time())

            for svc in TRACED_SERVICES:
                since_us = int(checkpoints.get(svc, 0))
                traces = fetch_new_traces(svc, since_us)
                max_seen_us = since_us

                for trace in traces:
                    if trace["traceID"] in seen_trace_ids:
                        continue
                    seen_trace_ids.add(trace["traceID"])

                    record = extract_record(trace, rolling_stats, system_snapshot)
                    if record is None:
                        continue

                    new_records.append(record)
                    trace_start_us = int(record["request_timestamp"] * 1_000_000)
                    max_seen_us = max(max_seen_us, trace_start_us)

                checkpoints[svc] = max_seen_us

            new_records.sort(key=lambda r: r["request_timestamp"])
            for record in new_records:
                outfile.write(json.dumps(record) + "\n")
            outfile.flush()

            total_written += len(new_records)
            save_checkpoints(args.checkpoint, checkpoints)

            elapsed = time.time() - poll_start
            print(f"[collector] poll #{poll_count}: +{len(new_records)} records (total={total_written}) in {elapsed:.1f}s")

            sleep_for = max(0, args.interval - elapsed)
            time.sleep(sleep_for)

    print(f"[collector] done. total records written: {total_written}")


if __name__ == "__main__":
    main()
