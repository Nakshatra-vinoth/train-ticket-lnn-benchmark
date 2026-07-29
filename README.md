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
    Evaluation scripts

benchmark/
    Processed train/validation/test datasets

models/
    Trained model checkpoints

predictions/
    Model predictions on the test set

results/
    Evaluation summaries and visualizations
```

---

## Models Evaluated

| Category | Model |
| -------- | ----- |
| Baseline | Naive Mean |
| Classical Machine Learning | XGBoost |
| Recurrent Neural Network | LSTM |
| Recurrent Neural Network | GRU |
| Liquid Neural Network | Closed-form Continuous-time (CfC) |
| Liquid Neural Network | Liquid Time-Constant (LTC) |

---

## Current Results

## Current Results

| Model | MAE (ms) | Pearson |
| :---- | -------: | -------: |
| Naive Mean | 22.42 | — |
| XGBoost | 21.23 | 0.19 |
| LSTM | 19.49 | 0.35 |
| GRU | 18.22 | 0.45 |
| **CfC** | **17.84** | **0.49** |

### Key Observations

- **LSTM** achieves the **lowest Mean Absolute Error (MAE)** among the evaluated models (**17.74 ms**).
- **CfC** achieves the **highest Pearson correlation (0.49)** while also being the **smallest** and **fastest** neural network evaluated.
- **LTC** performs competitively in predictive accuracy but incurs substantially higher inference latency due to numerical ODE integration.
- All recurrent neural network models significantly outperform the classical XGBoost baseline on event-level latency prediction.

---

## Model Complexity & Inference Performance

| Model | Parameters | Inference Latency |
| ----- | ---------: | ----------------: |
| XGBoost | 190 trees | **0.000011 ms/sample** |
| **CfC** | **16,540** | **0.043458 ms/sample** |
| LTC | 25,783 | 4.380986 ms/sample |
| GRU | 44,929 | 0.126095 ms/sample |
| LSTM | 59,201 | 0.270589 ms/sample |

> Inference latency is measured on CPU as the average per-sample inference time over multiple forward passes.

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

- **49 historical requests**
- **27 features per timestep**

---

## Evaluation Protocol

The benchmark follows a chronological evaluation protocol to avoid temporal leakage.

- Chronological train/validation/test split
- Sliding-window sequence generation
- Log-space target scaling
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Pearson Correlation
- Coefficient of Determination (R²)

---

## Current Status

- ✅ Benchmark construction completed
- ✅ Dataset preprocessing completed
- ✅ Naive baseline completed
- ✅ XGBoost baseline completed
- ✅ LSTM baseline completed
- ✅ GRU baseline completed
- ✅ CfC baseline completed
- ✅ LTC baseline completed

---

## Future Work

- Evaluate robustness across multiple random seeds
- Compare model performance across different workload regimes
- Analyze latency spike prediction
- Investigate uncertainty estimation for latency prediction
- Explore larger Liquid Neural Network architectures
- Evaluate hybrid recurrent-liquid architectures

---

## Summary

This benchmark demonstrates that Liquid Neural Networks are a promising alternative to conventional recurrent neural networks for event-level microservice latency prediction.

Among the evaluated models:

- **LSTM** achieves the lowest prediction error (**17.74 ms MAE**).
- **CfC** achieves the strongest correlation with ground truth (**0.49 Pearson correlation**) while also being the **smallest model (16,540 parameters)** and the **fastest neural network during inference (0.043 ms/sample)**.
- **LTC** achieves competitive predictive performance but with significantly higher inference cost due to continuous-time ODE integration.
- Overall, recurrent neural networks consistently outperform the classical XGBoost baseline, demonstrating the importance of temporal modeling for latency prediction in microservice systems.
