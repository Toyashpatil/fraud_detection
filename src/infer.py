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

FEATURE_COLS = [
    "amount","amount_log","balance","balance_to_amount","txns_24h","txns_7d",
    "is_weekend","hour_sin","hour_cos","unique_recipients","outbound_transfer_count",
    "recipient_rate","avg_amount_cust","amount_to_avg_ratio","unique_recipients_per_transfer",
    "base_risk","smurf_rule_flag","account_age_days"
]

def make_features(txn, account_agg):
    """
    txn: dict -> fields: amount (float), timestamp (datetime), txn_type (str)
    account_agg: dict -> precomputed per-account aggregates:
        unique_recipients, outbound_transfer_count, total_txn_count, avg_amount_cust,
        base_risk, account_age_days, balance, txns_24h, txns_7d
    """
    ts = txn.get("timestamp", datetime.utcnow())
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    hour = ts.hour
    weekday = ts.weekday()
    amount = float(txn["amount"])
    balance = float(account_agg.get("balance", 0.0))
    amount_log = math.log1p(amount)
    balance_to_amount = balance / (amount + 1.0)
    is_weekend = 1 if weekday in (5,6) else 0
    hour_sin = math.sin(2*math.pi*hour/24)
    hour_cos = math.cos(2*math.pi*hour/24)

    unique_recipients = account_agg.get("unique_recipients", 0)
    outbound_transfer_count = account_agg.get("outbound_transfer_count", 0)
    recipient_rate = outbound_transfer_count / (account_agg.get("total_txn_count", 0) + 1e-9)
    avg_amount_cust = account_agg.get("avg_amount_cust", 1.0)
    amount_to_avg_ratio = amount / (avg_amount_cust + 1e-9)
    unique_recipients_per_transfer = unique_recipients / (outbound_transfer_count + 1e-9)
    base_risk = account_agg.get("base_risk", 0.0)
    smurf_rule_flag = int((unique_recipients > 10) and (outbound_transfer_count > 15) and (amount < 200))
    account_age_days = account_agg.get("account_age_days", 0)
    txns_24h = account_agg.get("txns_24h", 0)
    txns_7d = account_agg.get("txns_7d", 0)

    row = [
        amount, amount_log, balance, balance_to_amount, txns_24h, txns_7d,
        is_weekend, hour_sin, hour_cos, unique_recipients, outbound_transfer_count,
        recipient_rate, avg_amount_cust, amount_to_avg_ratio, unique_recipients_per_transfer,
        base_risk, smurf_rule_flag, account_age_days
    ]
    return pd.Series(row, index=FEATURE_COLS)

def score_transaction(txn, account_agg, alpha=0.7):
    feats = make_features(txn, account_agg).values.reshape(1, -1)
    proba = rf.predict_proba(feats)[0,1]
    iso_score = -iso.decision_function(feats)[0]

    # normalize iso_score -> simple sigmoid mapping (tunable)
    iso_norm = 1 / (1 + np.exp(- (iso_score - 0.0)))

    fused_score = alpha * proba + (1-alpha) * iso_norm
    return {
        "supervised_proba": float(proba),
        "iso_score": float(iso_score),
        "iso_norm": float(iso_norm),
        "fused_score": float(fused_score)
    }
