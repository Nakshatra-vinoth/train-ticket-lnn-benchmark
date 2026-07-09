# Train Ticket LNN Benchmark

A benchmark for next-request latency prediction on the Train Ticket microservices application using conventional recurrent neural networks and Liquid Neural Networks (LNNs).

## Overview

This project builds an event-level benchmark from distributed traces and system metrics collected from the Train Ticket microservices benchmark.

The prediction task is:

> Given the previous 49 requests, predict the end-to-end latency of the next request.

The benchmark compares:

- LSTM
- GRU
- XGBoost
- Closed-form Continuous-time (CfC)
- Liquid Time-Constant (LTC)

## Pipeline

1. Generate workload using Locust
2. Collect traces from Jaeger
3. Collect system metrics from Prometheus/cAdvisor
4. Construct chronological event stream
5. Compute inter-arrival times (Δt)
6. Build sliding windows
7. Train and evaluate models

## Repository Structure

```
code/
    Benchmark construction
    Feature engineering
    Model training

data/
    Placeholder (generated locally)

models/
    Placeholder (trained locally)

predictions/
    Placeholder (generated locally)

results/
    Figures and evaluation plots
```

## Models

- LSTM
- GRU
- XGBoost
- CfC
- LTC

## Status

- Benchmark construction complete
- LSTM complete
- GRU complete
- XGBoost complete
- CfC complete
- LTC in progress

