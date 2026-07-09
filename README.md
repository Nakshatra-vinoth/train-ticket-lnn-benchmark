# Train Ticket LNN Benchmark

Event-level latency prediction benchmark for the Train Ticket microservices application using conventional recurrent neural networks and Liquid Neural Networks (LNNs).

---

## Overview

This repository contains the complete pipeline for constructing an event-level latency prediction benchmark from the Train Ticket microservices application.

The benchmark combines:

- Distributed traces collected from Jaeger
- System metrics collected from Prometheus/cAdvisor
- Multi-regime workloads generated using Locust

The prediction task is:

> Given the previous **49 requests**, predict the **end-to-end latency of the next request**.

---

## Benchmark Pipeline

```
Locust Workload
        │
        ▼
Train Ticket Microservices
        │
        ├────────► Jaeger Traces
        │
        └────────► Prometheus Metrics
                    │
                    ▼
           Feature Extraction
                    │
                    ▼
        Chronological Event Stream
                    │
                    ▼
        Sliding Window Generation
                    │
                    ▼
      Train / Validation / Test Split
                    │
                    ▼
          Model Training & Evaluation
```

---

## Repository Structure

```
code/
    Benchmark construction
    Feature engineering
    Model training scripts

data/
    Placeholder for generated datasets

models/
    Placeholder for trained models

predictions/
    Placeholder for model predictions

results/
    Evaluation figures
```

---

## Models Evaluated

| Category | Model |
|----------|-------|
| Baseline | Naive Mean |
| Classical ML | XGBoost |
| Recurrent Neural Network | LSTM |
| Recurrent Neural Network | GRU |
| Liquid Neural Network | Closed-form Continuous-time (CfC) |
| Liquid Neural Network | Liquid Time-Constant (LTC) |

---

## Current Results

| Model | MAE (ms) | Pearson Correlation |
|------|---------:|--------------------:|
| Naive Mean | 22.42 | — |
| XGBoost | 21.23 | 0.19 |
| LSTM | 19.49 | 0.35 |
| GRU | 18.22 | 0.45 |
| **CfC** | **17.84** | **0.49** |
| LTC | Training in progress | — |

CfC currently achieves the best overall performance among the completed models.

Compared to the strongest conventional recurrent baseline (GRU):

- MAE reduced from **18.22 ms** to **17.84 ms** (~2.1%)
- Pearson correlation improved from **0.45** to **0.49**

Compared to LSTM:

- MAE reduced by approximately **8.5%**

---

## Feature Set

Each request is represented using temporal, trace-level, and system-level features, including:

- Inter-arrival time (Δt)
- End-to-end latency
- Rolling request statistics
- Critical path latency
- Trace depth
- Number of spans
- Root service
- Services involved
- CPU usage
- Memory usage
- Network statistics

Each training sample consists of:

- 49 historical requests
- 27 features per timestep

---

## Evaluation Protocol

- Chronological train/validation/test split
- Sliding-window sequence generation
- Log-space target scaling
- Mean Absolute Error (MAE)
- Pearson correlation

---

## Current Status

- Benchmark construction completed
- Dataset preprocessing completed
- LSTM baseline completed
- GRU baseline completed
- XGBoost baseline completed
- CfC baseline completed
- LTC training in progress

---

## Future Work

- Complete LTC evaluation
- Compare performance across workload regimes
- Analyze latency spike prediction
- Evaluate statistical significance across multiple random seeds

