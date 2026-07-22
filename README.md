# Train Ticket LNN Benchmark

An event-level latency prediction benchmark for cloud-native microservices using conventional recurrent neural networks and Liquid Neural Networks (LNNs).

---

## Overview

This repository presents a complete benchmark for **end-to-end request latency prediction** on the **Train Ticket** microservices application. The benchmark integrates distributed tracing, infrastructure monitoring, and controlled workload generation to create a realistic dataset for sequential latency prediction.

The dataset is constructed from:

* **Distributed traces** collected using Jaeger
* **System-level resource metrics** collected from Prometheus and cAdvisor
* **Multi-regime workloads** generated with Locust to capture varying traffic conditions

The benchmark is designed to evaluate whether Liquid Neural Networks can effectively model temporal dynamics in production-like microservice environments compared with conventional deep learning and machine learning baselines.

---

## Prediction Task

Given the previous **49 requests** in chronological order, predict the **end-to-end latency of the next request**.

Each request is represented by temporal, trace-level, and system-level features, making this an **event-level sequential regression** problem.

---

## Benchmark Construction Pipeline

```text
Locust Workload Generator
           │
           ▼
 Train Ticket Microservices
           │
     ┌─────┴──────────┐
     ▼                ▼
Jaeger Traces   Prometheus Metrics
     │                │
     └──────┬─────────┘
            ▼
     Feature Extraction
            ▼
 Chronological Event Stream
            ▼
 Sliding Window Generation
            ▼
Chronological Train / Validation / Test Split
            ▼
     Model Training & Evaluation
```

---

## Repository Structure

```text
code/
    Dataset construction
    Feature engineering
    Benchmark generation
    Training scripts
    Evaluation scripts

data/
    Generated datasets
    Scaled feature matrices
    Target tensors

models/
    Saved model checkpoints

predictions/
    Model predictions
    Ground-truth targets

results/
    Training histories
    Evaluation reports
    Figures
    Benchmark summaries
```

---

## Dataset

### Sequence Configuration

| Property              |                Value |
| --------------------- | -------------------: |
| Sequence length       | 49 previous requests |
| Prediction target     | Next request latency |
| Features per timestep |                   27 |
| Task                  |           Regression |
| Split strategy        |        Chronological |

### Feature Categories

Each request contains information from multiple sources:

### Temporal Features

* Inter-arrival time (Δt)
* Previous request latency
* Rolling latency statistics
* Rolling request statistics

### Distributed Trace Features

* End-to-end latency
* Critical path latency
* Trace depth
* Number of spans
* Root service
* Services involved
* Request metadata

### System Metrics

* CPU utilization
* Memory utilization
* Network receive throughput
* Network transmit throughput
* Container-level resource statistics

---

## Models Evaluated

| Category                   | Model                             |
| -------------------------- | --------------------------------- |
| Baseline                   | Naive Mean                        |
| Classical Machine Learning | XGBoost                           |
| Recurrent Neural Network   | LSTM                              |
| Recurrent Neural Network   | GRU                               |
| Liquid Neural Network      | Closed-form Continuous-time (CfC) |
| Liquid Neural Network      | Liquid Time-Constant (LTC)        |

---

# Benchmark Results

| Model      |  MAE (ms) | Pearson Correlation |
| ---------- | --------: | ------------------: |
| Naive Mean |     22.42 |                   — |
| XGBoost    |     21.23 |                0.19 |
| GRU        |     18.22 |                0.45 |
| LTC        |     18.21 |                0.47 |
| **LSTM**   | **17.74** |                0.48 |
| CfC        |     17.84 |            **0.49** |

## Key Findings

* **LSTM** achieves the **lowest Mean Absolute Error (17.74 ms)**, making it the strongest model in terms of prediction accuracy.
* **CfC** achieves the **highest Pearson correlation (0.49)**, indicating the strongest ability to capture the overall latency trend.
* **LTC** performs competitively with both LSTM and GRU, demonstrating that Liquid Neural Networks are effective for event-level latency prediction.
* All recurrent neural network models significantly outperform the classical XGBoost baseline.
* The benchmark provides a reproducible comparison between conventional recurrent architectures and Liquid Neural Networks under identical training and evaluation conditions.

---

## Evaluation Methodology

The benchmark follows a strictly chronological evaluation protocol to prevent temporal leakage.

### Data Preparation

* Chronological ordering of requests
* Sliding-window sequence generation
* Train/validation/test chronological split
* Log-space target transformation
* Feature normalization using training statistics only

### Evaluation Metrics

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Pearson Correlation Coefficient
* Coefficient of Determination (R²)

---

## Current Status

| Component                   | Status |
| --------------------------- | :----: |
| Benchmark construction      |    ✅   |
| Dataset generation          |    ✅   |
| Feature engineering         |    ✅   |
| Train/Validation/Test split |    ✅   |
| XGBoost baseline            |    ✅   |
| LSTM baseline               |    ✅   |
| GRU baseline                |    ✅   |
| CfC baseline                |    ✅   |
| LTC baseline                |    ✅   |
| Benchmark evaluation        |    ✅   |

---

## Reproducibility

The repository contains everything required to reproduce the benchmark:

* Dataset construction pipeline
* Feature extraction scripts
* Chronological benchmark generation
* Baseline training implementations
* Saved model checkpoints
* Prediction outputs
* Evaluation reports
* Training histories

---

## Future Work

Potential extensions include:

* Multiple-seed evaluation for statistical robustness
* Workload regime-specific analysis
* Latency spike prediction
* Uncertainty-aware latency prediction
* Hyperparameter optimization for Liquid Neural Networks
* Transformer- and attention-based sequence models
* Cross-application evaluation on additional microservice benchmarks
