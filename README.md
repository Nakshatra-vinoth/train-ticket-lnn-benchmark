# Train Ticket LNN Benchmark

Event-level latency prediction benchmark for the Train Ticket microservices application using conventional recurrent neural networks and Liquid Neural Networks (LNNs).

---

## Overview

This repository presents a complete benchmark for **event-level latency prediction** in the Train Ticket microservices application. The benchmark combines distributed tracing, infrastructure monitoring, and workload generation to construct a chronological sequence prediction task.

The benchmark integrates:

- Distributed traces collected using **Jaeger**
- System metrics collected from **Prometheus** and **cAdvisor**
- Multi-regime workloads generated with **Locust**

The prediction task is:

> **Given the previous 49 requests, predict the end-to-end latency of the next request.**

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
    Model training and evaluation scripts

benchmark/
    Processed datasets

models/
    Trained model checkpoints

predictions/
    Test-set predictions

results/
    Evaluation summaries and visualizations
```

---

## Models Evaluated

| Category | Model |
|----------|-------|
| Baseline | Naive Mean |
| Classical Machine Learning | XGBoost |
| Recurrent Neural Network | LSTM |
| Recurrent Neural Network | GRU |
| Liquid Neural Network | Closed-form Continuous-time (CfC) |
| Liquid Neural Network | Liquid Time-Constant (LTC) |

---

## Current Results

| Model | MAE (ms) | Pearson |
| :---- | -------: | -------: |
| Naive Mean | 22.42 | — |
| XGBoost | 21.23 | 0.19 |
| LSTM | 19.49 | 0.35 |
| GRU | 18.22 | 0.45 |
| LTC | 18.21 | 0.47 |
| **CfC** | **17.84** | **0.49** |

### Key Observations

- **CfC** achieves the **lowest Mean Absolute Error (17.84 ms)** and the **highest Pearson correlation (0.49)** among all evaluated models.
- Both **CfC** and **LTC** are based on the same underlying continuous-time liquid dynamics. The key difference lies in how these dynamics are computed: **CfC uses a closed-form approximation**, whereas **LTC numerically solves the underlying ordinary differential equation (ODE) at every timestep**.
- Both liquid neural architectures improve the correlation between predicted and ground-truth latency compared to conventional recurrent networks. However, **only the closed-form CfC model translates this stronger temporal modeling into a reduction in absolute prediction error**.
- Although **LTC** achieves a higher Pearson correlation than the GRU and LSTM baselines, its substantially higher computational cost is **not accompanied by an improvement in prediction accuracy over CfC**.
- Overall, **CfC provides the best trade-off between predictive accuracy, model size, and inference efficiency** on this benchmark.

---

## Model Complexity & Inference Performance

| Model | Parameters | Inference Latency |
| ----- | ---------: | ----------------: |
| XGBoost | 190 trees | **0.000011 ms/sample** |
| **CfC** | **16,540** | **0.043458 ms/sample** |
| LTC | 25,783 | 4.380986 ms/sample |
| GRU | 44,929 | 0.126095 ms/sample |
| LSTM | 59,201 | 0.270589 ms/sample |

> Inference latency was measured on CPU as the average per-sample inference time over multiple forward passes.

---

## Feature Set

Each request is represented using temporal, trace-level, and system-level information, including:

- Inter-arrival time (Δt)
- End-to-end latency
- Rolling request statistics
- Critical path latency
- Trace depth
- Number of spans
- Root service
- Services involved
- CPU utilization
- Memory utilization
- Network statistics

Each training sample consists of:

- **49 historical requests**
- **27 features per timestep**

---

## Evaluation Protocol

Models are evaluated using a chronological split to prevent temporal leakage.

Evaluation includes:

- Chronological train/validation/test split
- Sliding-window sequence generation
- Log-space target scaling
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Pearson correlation coefficient
- Coefficient of determination (R²)

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
- Analyze performance across different workload regimes
- Investigate latency spike prediction
- Explore uncertainty estimation for latency prediction
- Evaluate larger Liquid Neural Networks
- Explore hybrid recurrent-liquid architectures

---

## Summary

This benchmark demonstrates that Liquid Neural Networks are an effective approach for event-level latency prediction in microservice systems.

Among the evaluated models, **CfC** achieves the best overall performance, obtaining the **lowest MAE (17.84 ms)** and the **highest Pearson correlation (0.49)** while also being the **smallest neural network (16,540 trainable parameters)** and the **fastest neural architecture during inference (0.043458 ms/sample)**.

Although **LTC** is derived from the same continuous-time dynamics, its reliance on numerical ODE integration results in significantly higher inference latency (**4.38 ms/sample**) without improving predictive accuracy over CfC. This suggests that the closed-form approximation used by CfC retains the benefits of liquid dynamics while offering substantially better computational efficiency.

Overall, the results indicate that **CfC provides the strongest balance between accuracy, correlation, model complexity, and inference speed**, making it the most effective neural architecture evaluated for this benchmark.
