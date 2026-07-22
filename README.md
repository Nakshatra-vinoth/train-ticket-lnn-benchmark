# Train Ticket LNN Benchmark

Event-level latency prediction benchmark for the Train Ticket microservices application using conventional recurrent neural networks and Liquid Neural Networks (LNNs).

---

## Overview

This repository contains the complete pipeline for constructing an event-level latency prediction benchmark from the Train Ticket microservices application.

The benchmark combines:

* Distributed traces collected from Jaeger
* System metrics collected from Prometheus/cAdvisor
* Multi-regime workloads generated using Locust

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
    Generated datasets and preprocessing artifacts

models/
    Trained model checkpoints

predictions/
    Model predictions on the test set

results/
    Evaluation reports and visualizations
```

---

## Models Evaluated

| Category                 | Model                             |
| ------------------------ | --------------------------------- |
| Baseline                 | Naive Mean                        |
| Classical ML             | XGBoost                           |
| Recurrent Neural Network | LSTM                              |
| Recurrent Neural Network | GRU                               |
| Liquid Neural Network    | Closed-form Continuous-time (CfC) |
| Liquid Neural Network    | Liquid Time-Constant (LTC)        |

---

## Current Results

| Model      |  MAE (ms) | Pearson Correlation |
| ---------- | --------: | ------------------: |
| Naive Mean |     22.42 |                   — |
| XGBoost    |     21.23 |                0.19 |
| GRU        |     18.22 |                0.45 |
| LTC        |     18.21 |                0.47 |
| LSTM       |     17.74 |                0.48 |
| **CfC**    | **17.84** |            **0.49** |

### Key Observations

* **LSTM** achieves the **lowest Mean Absolute Error (MAE)** among the evaluated models (**17.74 ms**).
* **CfC** achieves the **highest Pearson correlation (0.49)**, indicating the strongest linear agreement with the true latency values.
* **LTC** performs competitively with conventional recurrent models, achieving **18.21 ms MAE** and **0.47 Pearson correlation**, demonstrating that liquid neural architectures are viable for event-level microservice latency prediction.
* Overall, all recurrent neural network models substantially outperform the classical XGBoost baseline.

---

## Feature Set

Each request is represented using temporal, trace-level, and system-level features, including:

* Inter-arrival time (Δt)
* End-to-end latency
* Rolling request statistics
* Critical path latency
* Trace depth
* Number of spans
* Root service
* Services involved
* CPU usage
* Memory usage
* Network statistics

Each training sample consists of:

* 49 historical requests
* 27 features per timestep

---

## Evaluation Protocol

* Chronological train/validation/test split
* Sliding-window sequence generation
* Log-space target scaling
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Pearson correlation
* Coefficient of determination (R²)

---

## Current Status

* ✅ Benchmark construction completed
* ✅ Dataset preprocessing completed
* ✅ XGBoost baseline completed
* ✅ LSTM baseline completed
* ✅ GRU baseline completed
* ✅ CfC baseline completed
* ✅ LTC baseline completed

---

## Future Work

* Evaluate robustness across multiple random seeds
* Compare model performance across workload regimes
* Analyze latency spike prediction
* Investigate uncertainty estimation for latency prediction
* Explore larger Liquid Neural Network architectures and hybrid models
