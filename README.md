# Fraud Detection — Smurfing / Structuring Detector

This repository contains a prototype pipeline for detecting smurfing and structuring (money-laundering) behavior in transaction data. It includes synthetic data generation, injection of smurf/structuring scenarios, feature engineering (including short-window burst and recipient-distribution features), a hybrid detection approach (supervised RandomForest + IsolationForest), and utilities for inference and evaluation.

This README summarizes the project structure, how to run the pipeline locally, and next steps for making the system production-ready.

## Key Features
- Synthetic dataset generation with configurable injectors (`src/data_generator.py`, `src/smurf_injector.py`) that create labeled smurfing and structuring examples.
- Feature engineering (`src/feature_engineering.py`) with:
  - Transaction features (amount, hour/cyclical encodings, log-amount).
  - Account-level aggregates and rolling-window stats (24h sums/medians/max).
  - Short-window burst counters (1h/3h/6h) and peak-per-hour features.
  - Recipient-distribution features (recipient entropy, unique recipients in windows, new-recipient fraction).
- Graph features utilities (`src/graph_features.py`) to compute lightweight bipartite customer→recipient signals (degree, shared recipients, component size).
- Training pipeline with persisted artifacts: `models/feature_pipeline.pkl`, `models/iso_smurf_model.pkl`, and `models/feature_list.json` (created by `src/run_pipeline.py`).
- Inference helpers (`src/infer.py`): `score_transaction` (single-transaction scorer using persisted pipeline) and `score_stream` (computes short-window aggregates from account history and scores). 

## Repo Layout
- `src/` — main source code
  - `data_generator.py` — create base synthetic transactions
  - `smurf_injector.py` — injects smurfing/structuring cases (now tags `fraud_case_id` and `account_label`)
  - `feature_engineering.py` — create_features(...) builds X, y and added burst/recipient features
  - `model_training.py` — training helpers and persistence (pipeline + iso model)
  - `infer.py` — scoring helpers and streaming scorer
  - `graph_features.py` — build bipartite graph and compute graph features
  - `run_pipeline.py` — end-to-end orchestration: generate → inject → features → train → evaluate → save artifacts
- `models/` — trained artifacts (created by `run_pipeline.py`)
- `data/` — generated synthetic data and evaluation outputs (PR curve, `top_alerts.csv`)
- `tests/` — test scripts (small smoke tests)
- `requirements.txt` — dependencies (add `networkx` for graph utilities)

## Getting Started (Local)
1. Create a virtual environment and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the full pipeline (this will generate synthetic data, inject patterns, train models, and save artifacts):

```powershell
python .\src\run_pipeline.py
```

Artifacts saved by the pipeline:
- `models/feature_pipeline.pkl` — scikit-learn Pipeline (StandardScaler + RandomForest)
- `models/iso_smurf_model.pkl` — IsolationForest model
- `models/feature_list.json` — canonical feature order used for inference
- `data/synthetic_smurf_transactions.csv` — synthetic dataset with labels and features
- `data/pr_curve.png`, `data/top_alerts.csv` — evaluation outputs

3. Score a single transaction example using `score_one.py`:

```powershell
python .\score_one.py
```

4. Use `score_stream` to score a transaction with its recent account history (example available in `tests/tmp_score_stream_test.py`). To run tests/scripts that import the `src` package, set `PYTHONPATH` or run from the project root:

```powershell
$env:PYTHONPATH = Get-Location
python .\tests\tmp_score_stream_test.py
```

## Development Notes & Next Steps
- Tests: add unit tests for `feature_engineering` and `graph_features`, plus an integration test for `run_pipeline.py` on a small dataset.
- Graph integration: `src/graph_features.py` is available; integrate its outputs into `create_features` and retrain to measure lift (postponed by default).
- Production hardening:
  - Persist feature schema and add compatibility checks at inference.
  - Add calibration and threshold-selection logic using Precision@K or acceptable false positive rates.
  - Replace NetworkX with sparse / scalable graph processing if dataset gets large.
  - Add monitoring (prediction volume, alert rate, feature drift).

## Design Rationale
- Hybrid approach (rules + supervised + unsupervised) provides robust coverage: fast rule pre-filters, a supervised model trained on injected patterns, and an unsupervised novelty detector for unseen patterns.
- Short-window bursts and recipient entropy are high-ROI signals for smurfing. Graph features help find coordinated multi-account campaigns.

## Troubleshooting
- Import errors like `ModuleNotFoundError: No module named 'src'`: ensure `PYTHONPATH` includes the repository root or run Python from the repo root; alternatively, add the project root to `sys.path` in test scripts (temporary) or install the package in editable mode.

## Contact / Attribution
This project is a prototype. For production use, ensure legal and compliance review, data governance, and proper model validation.

---

If you want, I can now:
- Integrate graph features into `create_features` and retrain models (prototype + evaluation), or
- Add unit tests and CI workflow, or
- Add README sections for deployment and artifact versioning.

Tell me which of the above to proceed with next.
