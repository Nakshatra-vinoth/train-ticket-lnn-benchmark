"""
Multi-regime workload generator for Train Ticket latency dataset collection.
See class/function docstrings for design rationale.
Run with: locust -f locustfile.py --host http://localhost:8080 --headless
"""

import random
import time
import os
from locust import HttpUser, task, LoadTestShape, events
from datetime import datetime, timedelta

# Train Ticket seed data is typically dated relative to when the DB was seeded.
# Use "tomorrow" from today to stay safely inside the seeded window.
TARGET_DEPARTURE_DATE = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

REGIME_PHASES = [
    ("low_load",     180,  3,  1),
    ("ramp_up",      180, 25,  3),
    ("steady_state", 180, 25,  2),
    ("bursty",       180, 60, 10),
    ("congestion",   180, 80,  8),
    ("recovery",     180,  5,  5),
]

CYCLE_DURATION = sum(p[1] for p in REGIME_PHASES)

REGIME_LAMBDA = {
    "low_load":     0.3,
    "ramp_up":      0.8,
    "steady_state": 1.0,
    "bursty":       2.5,
    "congestion":   3.0,
    "recovery":     0.5,
}

BURST_PROBABILITY = 0.07
PARETO_SHAPE = 1.5
PARETO_SCALE_SEC = 0.05

_START_TIME = time.time()


def current_regime(elapsed_seconds):
    t = elapsed_seconds % CYCLE_DURATION
    acc = 0
    for name, duration, _, _ in REGIME_PHASES:
        acc += duration
        if t < acc:
            return name
    return REGIME_PHASES[-1][0]


class TrainTicketShape(LoadTestShape):
    TOTAL_RUN_SECONDS = int(os.environ.get("LOCUST_TOTAL_RUN_SECONDS", CYCLE_DURATION * 4))

    def tick(self):
        run_time = self.get_run_time()
        if run_time > self.TOTAL_RUN_SECONDS:
            return None
        t = run_time % CYCLE_DURATION
        acc = 0
        for name, duration, users, spawn_rate in REGIME_PHASES:
            acc += duration
            if t < acc:
                return (users, spawn_rate)
        return (REGIME_PHASES[-1][2], REGIME_PHASES[-1][3])


def regime_aware_wait():
    regime = current_regime(time.time() - _START_TIME)
    lam = REGIME_LAMBDA[regime]
    if random.random() < BURST_PROBABILITY:
        gap = (random.paretovariate(PARETO_SHAPE)) * PARETO_SCALE_SEC
    else:
        gap = random.expovariate(lam)
    return min(gap, 30.0)


@events.test_start.add_listener
def _on_test_start(environment, **kwargs):
    global _START_TIME
    _START_TIME = time.time()
    print(f"[workload] Test started. Cycle duration={CYCLE_DURATION}s, "
          f"total run={TrainTicketShape.TOTAL_RUN_SECONDS}s "
          f"(~{TrainTicketShape.TOTAL_RUN_SECONDS / CYCLE_DURATION:.1f} cycles)")


STATIONS = [
    ("Shang Hai", "Su Zhou"),
    ("Shang Hai", "Nan Jing"),
    ("Su Zhou", "Shang Hai"),
    ("Nan Jing", "Shang Hai"),
]


class TrainTicketUser(HttpUser):
    wait_time = lambda self: regime_aware_wait()

    @task(10)
    def search_trips_travel(self):
        start, end = random.choice(STATIONS)
        self.client.post(
            "/api/v1/travelservice/trips/left",
            json={"startingPlace": start, "endPlace": end, "departureTime": TARGET_DEPARTURE_DATE},
            name="/api/v1/travelservice/trips/left",
        )

    @task(6)
    def search_trips_travel2(self):
        start, end = random.choice(STATIONS)
        self.client.post(
            "/api/v1/travel2service/trips/left",
            json={"startingPlace": start, "endPlace": end, "departureTime": TARGET_DEPARTURE_DATE},
            name="/api/v1/travel2service/trips/left",
        )
