import pickle
import pandas as pd
import math
import numpy as np
from datetime import datetime

# Load models (paths match run_pipeline outputs)
RF_PATH = "models/rf_smurf_model.pkl"
ISO_PATH = "models/iso_smurf_model.pkl"

with open(RF_PATH, "rb") as f:
    rf = pickle.load(f)
with open(ISO_PATH, "rb") as f:
    iso = pickle.load(f)

# Feature column order must match feature_engineering.py
FEATURE_COLS = [
    "amount","amount_log","balance","balance_to_amount","txns_24h","txns_7d",
    "is_weekend","hour_sin","hour_cos","unique_recipients","outbound_transfer_count",
    "recipient_rate","avg_amount_cust","amount_to_avg_ratio","unique_recipients_per_transfer",
    "base_risk","smurf_rule_flag","account_age_days",
    # new structuring features
    "last_inflow_amount","time_since_last_inflow_hours","sum_outflow_24h","count_outflow_24h",
    "median_outflow_24h","max_outflow_24h","inflow_to_outflow_ratio_24h","is_large_inflow_recent",
    "num_transfers_above_20k_24h"
]

def make_features(txn, account_agg):
    """
    txn: dict with transaction-level fields (amount, timestamp, txn_type, txns_24h, txns_7d, balance)
    account_agg: dict with precomputed per-account aggregates (keys optional)
    Returns: pandas.Series with FEATURE_COLS
    """
    ts = txn.get("timestamp", datetime.utcnow())
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    hour = ts.hour
    weekday = ts.weekday()

    amount = float(txn.get("amount", account_agg.get("amount", 0.0)))
    balance = float(account_agg.get("balance", 0.0))
    amount_log = math.log1p(amount)
    balance_to_amount = balance / (amount + 1.0)
    is_weekend = 1 if weekday in (5,6) else 0
    hour_sin = math.sin(2*math.pi*hour/24)
    hour_cos = math.cos(2*math.pi*hour/24)

    # existing account aggregates (safe defaults if missing)
    unique_recipients = float(account_agg.get("unique_recipients", 0.0))
    outbound_transfer_count = float(account_agg.get("outbound_transfer_count", 0.0))
    recipient_rate = float(outbound_transfer_count) / (account_agg.get("total_txn_count", 0.0) + 1e-9)
    avg_amount_cust = float(account_agg.get("avg_amount_cust", amount if amount>0 else 1.0))
    amount_to_avg_ratio = amount / (avg_amount_cust + 1e-9)
    unique_recipients_per_transfer = unique_recipients / (outbound_transfer_count + 1e-9)
    base_risk = float(account_agg.get("base_risk", 0.0))
    smurf_rule_flag = int((unique_recipients > 10) and (outbound_transfer_count > 15) and (amount < 200))
    account_age_days = float(account_agg.get("account_age_days", 0.0))
    txns_24h = float(account_agg.get("txns_24h", txn.get("txns_24h", 0.0)))
    txns_7d = float(account_agg.get("txns_7d", txn.get("txns_7d", 0.0)))

    # new structuring features (safe defaults)
    last_inflow_amount = float(account_agg.get("last_inflow_amount", 0.0))
    time_since_last_inflow_hours = float(account_agg.get("time_since_last_inflow_hours", 9999.0))
    sum_outflow_24h = float(account_agg.get("sum_outflow_24h", 0.0))
    count_outflow_24h = float(account_agg.get("count_outflow_24h", 0.0))
    median_outflow_24h = float(account_agg.get("median_outflow_24h", 0.0))
    max_outflow_24h = float(account_agg.get("max_outflow_24h", 0.0))
    inflow_to_outflow_ratio_24h = float(account_agg.get("inflow_to_outflow_ratio_24h", 0.0))
    is_large_inflow_recent = int(account_agg.get("is_large_inflow_recent", 0))
    num_transfers_above_20k_24h = float(account_agg.get("num_transfers_above_20k_24h", 0.0))

    row = [
        amount, amount_log, balance, balance_to_amount, txns_24h, txns_7d,
        is_weekend, hour_sin, hour_cos, unique_recipients, outbound_transfer_count,
        recipient_rate, avg_amount_cust, amount_to_avg_ratio, unique_recipients_per_transfer,
        base_risk, smurf_rule_flag, account_age_days,
        last_inflow_amount, time_since_last_inflow_hours, sum_outflow_24h, count_outflow_24h,
        median_outflow_24h, max_outflow_24h, inflow_to_outflow_ratio_24h, is_large_inflow_recent,
        num_transfers_above_20k_24h
    ]
    return pd.Series(row, index=FEATURE_COLS)

def score_transaction(txn, account_agg, alpha=0.7):
    """
    Compute supervised proba, unsupervised anomaly, normalized iso score, and fused score.
    If account_agg includes structuring-rule conditions, you may choose to override fused_score externally.
    """
    feats = make_features(txn, account_agg).values.reshape(1, -1)
    proba = rf.predict_proba(feats)[0,1]
    iso_score = -iso.decision_function(feats)[0]
    iso_norm = 1 / (1 + np.exp(- (iso_score - 0.0)))  # sigmoid normalization

    fused_score = alpha * proba + (1-alpha) * iso_norm

    return {
        "supervised_proba": float(proba),
        "iso_score": float(iso_score),
        "iso_norm": float(iso_norm),
        "fused_score": float(fused_score)
    }
