# Train Ticket LNN Benchmark

Event-level latency prediction benchmark for the Train Ticket microservices application, comparing a naive baseline, XGBoost, discrete-time recurrent networks (LSTM, GRU), and continuous-time Liquid Neural Networks (LTC, CfC).

---

## 1. Overview

This repository builds an **event-level latency prediction benchmark** from the Train Ticket microservices application under a multi-regime Locust workload, and benchmarks six models on it:

> **Given the 49 most recent requests (27 features each), predict the end-to-end latency of the next request.**

Pipeline: Locust workload → Train Ticket → Jaeger traces + Prometheus/cAdvisor metrics → feature extraction (27 features) → chronological sliding-window split (w=49) → model training & evaluation.

| Model | MAE (ms) | Pearson r |
|---|---:|---:|
| Naive Mean | 22.42 | — |
| XGBoost | 21.23 | 0.19 |
| LSTM | 19.49 | 0.35 |
| GRU | 18.22 | 0.45 |
| LTC | 18.21 | 0.47 |
| **CfC** | **17.84** | **0.49** |

Full discussion of these results is in the report. This README covers how to reproduce them.

---

## 2. Repository Structure

```
code/
  collector.py              # Live data collection from Jaeger/Prometheus (needs a running Train Ticket + docker_stats_collector.py, see §6)
  compute_deltat.py         # dataset.jsonl -> dataset_with_deltat.jsonl
  construct_benchmark.py    # dataset_with_deltat.jsonl -> windows_index.jsonl (chronological, gap-purged)
  split_windows.py          # windows_index.jsonl -> windows_{train,val,test}.jsonl
  extract_features.py       # -> X_{train,val,test}.npy, y_{train,val,test}.npy (27-dim features, raw ms target)
  scale_features.py         # -> X_{train,val,test}_scaled.npy (z-score, fit on train only)
  scale_targets_log.py      # -> y_{train,val,test}_scaled.npy (log1p + standardize, fit on train only)
  locustfile.py             # Multi-regime workload generator (6 phases: low_load, ramp_up, steady_state, bursty, congestion, recovery)

  train_xgboost_baseline.py train_lstm_baseline.py train_gru_baseline.py
  train_ltc_baseline.py     train_cfc_baseline.py       # <- the FIVE scripts that produced Table 1/2 
  train_cfc_v2.py train_cfc_v4.py                        # Exploratory CfC ablations, NOT used for the reported numbers 

  evaluate_xgboost.py evaluate_lstm.py evaluate_gru.py
  evaluate_ltc.py evaluate_cfc.py                         # Reload the saved checkpoints, report params + CPU inference latency (Table 2)

benchmark/
  scaler_params.json         # Versioned. The rest of this folder's contents (npy/jsonl) are gitignored — download separately, see §4

models/                      # Created when you run training — holds *_best.pt checkpoints (gitignored)
xgboost_baseline.json        # Trained XGBoost booster, checked into the repo root (the only pre-trained artifact included)
requirements.txt
```

Note: `models/`, `predictions/`, `results/` are **not** pre-populated — they're produced by running the scripts, and everything under them is gitignored.

---

## 3. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Key packages: `torch`, `ncps` (LTC/CfC implementations), `xgboost`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `locust`.

All neural models except CfC's training script are forced to `torch.device("cpu")`, and inference is measured with `torch.set_num_threads(1)`, matching the CPU inference numbers in Table 2. A GPU is not required to reproduce any result in this repo.

---

## 4. Getting the Data (Required)

The processed benchmark artifacts (frozen dataset, windows, and train/val/test `.npy` arrays) are too large for git and are provided separately via Google Drive:

**Drive link: `https://drive.google.com/drive/folders/1f_ZwlAr9vOFgPFIbXurUNZlManji0AMW?usp=sharing`**

Download every file shown there and place them in `benchmark/`, so it contains:

```
benchmark/
  dataset_with_deltat.jsonl
  windows_train.jsonl  windows_val.jsonl  windows_test.jsonl
  X_train.npy  X_val.npy  X_test.npy
  X_train_scaled.npy  X_val_scaled.npy  X_test_scaled.npy
  y_train.npy  y_val.npy  y_test.npy
  y_train_scaled.npy  y_val_scaled.npy  y_test_scaled.npy
  scaler_params.json
```

This is the frozen, already-split, already-scaled benchmark used for every number in Table 1 and Table 2. **You do not need to regenerate these files to reproduce the paper's results** — only to reproduce the underlying data-collection process itself (§6, not required for grading).

---

## 5. Reproducing Table 1 and Table 2 (Recommended Path)

All five training scripts read data via a path relative to their own file location (`../benchmark`), so they work regardless of your current directory — **except** that each script saves its checkpoint under a bare filename (e.g. `cfc_baseline_best.pt`), and the `evaluate_*.py` scripts expect to find that checkpoint in `models/`. So: run training from inside `models/`.

```bash
mkdir -p models
cd models

python ../code/train_xgboost_baseline.py   # -> ../models/ (xgboost_baseline.json is bare-saved here too;
                                            #    move/copy it to the repo root to match evaluate_xgboost.py, see note below)
python ../code/train_lstm_baseline.py      # -> lstm_baseline_best.pt
python ../code/train_gru_baseline.py       # -> gru_baseline_best.pt
python ../code/train_ltc_baseline.py       # -> ltc_baseline_best.pt
python ../code/train_cfc_baseline.py       # -> cfc_baseline_best.pt  

cd ..
python code/evaluate_xgboost.py   # run from repo root: evaluate_xgboost.py loads "xgboost_baseline.json" from the cwd
python code/evaluate_lstm.py
python code/evaluate_gru.py
python code/evaluate_ltc.py
python code/evaluate_cfc.py
```

Each `train_*.py` script prints test-set MAE, RMSE, MAPE, R², and Pearson r (Table 1), and each `evaluate_*.py` script prints trainable parameter count and CPU inference latency per sample (Table 2).

## 6. Full Pipeline From Raw Traces (Optional)

For transparency, the full collection pipeline is:

```
locust -f code/locustfile.py --host http://<train-ticket-host>:8080 --headless
  → python code/collector.py --interval 30 --duration-hours 12 --output dataset.jsonl
  → python code/compute_deltat.py            # dataset.jsonl -> dataset_with_deltat.jsonl
  → python code/construct_benchmark.py       # -> windows_index.jsonl (see path caveat in §5)
  → python code/split_windows.py             # -> windows_{train,val,test}.jsonl
  → python code/extract_features.py          # -> X_*.npy, y_*.npy
  → python code/scale_features.py            # -> X_*_scaled.npy
  → python code/scale_targets_log.py         # -> y_*_scaled.npy, updates scaler_params.json
```

This requires a live Train Ticket deployment (with Jaeger and Prometheus/cAdvisor exposed) and several hours of Locust traffic, and re-running it will **not** reproduce the benchmark bit-for-bit — request timing, latency, and system-metric values depend on live infrastructure load at collection time. `collector.py` also imports `docker_stats_collector.get_container_stats`, which is not included in this repository and must be supplied separately. This section is included for transparency about how the benchmark was built, not as a required or practically reproducible step — use the frozen dataset in §4 for actual reproduction.

---

## 8. Models Evaluated

| Category | Model |
|---|---|
| Baseline | Naive Mean |
| Classical ML | XGBoost |
| Recurrent Neural Network | LSTM |
| Recurrent Neural Network | GRU |
| Liquid Neural Network | Liquid Time-Constant (LTC) |
| Liquid Neural Network | Closed-form Continuous-time (CfC) |

## 9. Feature Set

27 features per timestep, over a 49-step window:
- Temporal: inter-arrival time (Δt), rolling latency statistics
- Trace topology: trace depth, span count
- Service identity: root service (one-hot), services involved (multi-hot)
- System-level: CPU, memory, and network utilization per service

## 10. Evaluation Protocol

- Chronological train/val/test split (70/15/15) with a 50-window purge zone at each boundary to prevent leakage
- Log-space target scaling (log1p + standardize) for all five learned models; see §4 for the frozen scaled targets
- Metrics: MAE, RMSE, MAPE, R², Pearson r — all reported in real milliseconds after inverse-transforming predictions
