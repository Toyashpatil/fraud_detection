import joblib
import pandas as pd
import math
import numpy as np
from datetime import datetime
import json

# Load persisted artifacts produced by run_pipeline
PIPELINE_PATH = "models/feature_pipeline.pkl"
ISO_PATH = "models/iso_smurf_model.pkl"
FEATURE_LIST_PATH = "models/feature_list.json"

try:
    pipeline = joblib.load(PIPELINE_PATH)
except Exception:
    pipeline = None

try:
    iso = joblib.load(ISO_PATH)
except Exception:
    iso = None

try:
    with open(FEATURE_LIST_PATH, "r") as f:
        FEATURE_COLS = json.load(f)
except Exception:
    # fallback to a conservative default (keeps compatibility)
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

    # Build row according to FEATURE_COLS; missing keys use safe defaults
    row = {
        "amount": amount,
        "amount_log": amount_log,
        "balance": balance,
        "balance_to_amount": balance_to_amount,
        "txns_24h": txns_24h,
        "txns_7d": txns_7d,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "unique_recipients": unique_recipients,
        "outbound_transfer_count": outbound_transfer_count,
        "recipient_rate": recipient_rate,
        "avg_amount_cust": avg_amount_cust,
        "amount_to_avg_ratio": amount_to_avg_ratio,
        "unique_recipients_per_transfer": unique_recipients_per_transfer,
        "base_risk": base_risk,
        "smurf_rule_flag": smurf_rule_flag,
        "account_age_days": account_age_days,
        "last_inflow_amount": last_inflow_amount,
        "time_since_last_inflow_hours": time_since_last_inflow_hours,
        "sum_outflow_24h": sum_outflow_24h,
        "count_outflow_24h": count_outflow_24h,
        "median_outflow_24h": median_outflow_24h,
        "max_outflow_24h": max_outflow_24h,
        "inflow_to_outflow_ratio_24h": inflow_to_outflow_ratio_24h,
        "is_large_inflow_recent": is_large_inflow_recent,
        "num_transfers_above_20k_24h": num_transfers_above_20k_24h
    }

    # ensure any newly added features are present in row with defaults
    series = pd.Series({k: float(row.get(k, 0.0)) for k in FEATURE_COLS})
    return series

def score_transaction(txn, account_agg, alpha=0.7):
    """
    Compute supervised proba, unsupervised anomaly, normalized iso score, and fused score.
    If account_agg includes structuring-rule conditions, you may choose to override fused_score externally.
    """
    feats = make_features(txn, account_agg).values.reshape(1, -1)

    # supervised probability via persisted pipeline if available
    if pipeline is not None:
        proba = pipeline.predict_proba(feats)[0, 1]
    else:
        # fallback: if pipeline not available, try to use iso only
        proba = 0.0

    iso_score = None
    iso_norm = 0.0
    if iso is not None:
        iso_score = -iso.decision_function(feats)[0]
        iso_norm = 1 / (1 + np.exp(- (iso_score - 0.0)))  # sigmoid normalization

    fused_score = alpha * proba + (1-alpha) * iso_norm

    return {
        "supervised_proba": float(proba),
        "iso_score": float(iso_score) if iso_score is not None else None,
        "iso_norm": float(iso_norm),
        "fused_score": float(fused_score)
    }


def score_stream(txn, account_history_df, alpha=0.7, alert_threshold=None):
    """
    Compute rolling/short-window aggregates from an account's historical transactions and score a new txn.

    txn: dict with transaction-level fields (amount, timestamp, txn_type, recipient_account_id)
    account_history_df: pandas.DataFrame of prior transactions for the same customer (may include the txn itself)
    alpha: fusion weight for supervised model
    alert_threshold: optional threshold for fused_score to return an `alert` boolean

    Returns: dict with supervised_proba, iso_score, iso_norm, fused_score, alert (if threshold provided), and account_agg used.
    """
    import pandas as pd
    # Make a defensive copy and ensure timestamp parsing
    hist = account_history_df.copy()
    if "timestamp" in hist.columns:
        hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    else:
        hist["timestamp"] = pd.NaT

    # Align reference time to txn timestamp
    ts = txn.get("timestamp")
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)
    if ts is None:
        ts = pd.Timestamp.utcnow()

    # Filter history to records up to txn time
    hist = hist[hist["timestamp"] <= ts]

    # Basic aggregates
    unique_recipients = int(hist[hist["txn_type"].isin(["transfer","debit"])]["recipient_account_id"].nunique()) if "recipient_account_id" in hist.columns else 0
    outbound_transfer_count = int((hist["txn_type"]=="transfer").sum())
    total_txn_count = int(len(hist))
    avg_amount_cust = float(hist["amount"].mean()) if "amount" in hist.columns and len(hist)>0 else float(txn.get("amount", 0.0))
    base_risk = float(hist["base_risk"].iloc[-1]) if "base_risk" in hist.columns and len(hist)>0 else float(0.0)
    account_age_days = float(hist["account_age_days"].iloc[-1]) if "account_age_days" in hist.columns and len(hist)>0 else 0.0
    balance = float(hist["balance"].iloc[-1]) if "balance" in hist.columns and len(hist)>0 else float(txn.get("balance", 0.0))

    # Time-window aggregates helper
    def window_agg(df, window):
        start = ts - pd.Timedelta(window)
        return df[(df["timestamp"] > start) & (df["timestamp"] <= ts)]

    w_1h = window_agg(hist, "1h")
    w_3h = window_agg(hist, "3h")
    w_6h = window_agg(hist, "6h")
    w_24h = window_agg(hist, "24h")
    w_7d = window_agg(hist, "7d")

    sum_outflow_24h = float(w_24h[ w_24h["txn_type"].isin(["transfer","debit"]) ]["amount"].sum()) if "amount" in hist.columns else 0.0
    count_outflow_24h = int(w_24h[ w_24h["txn_type"].isin(["transfer","debit"]) ].shape[0])
    median_outflow_24h = float(w_24h[ w_24h["txn_type"].isin(["transfer","debit"]) ]["amount"].median()) if len(w_24h)>0 else 0.0
    max_outflow_24h = float(w_24h[ w_24h["txn_type"].isin(["transfer","debit"]) ]["amount"].max()) if len(w_24h)>0 else 0.0
    num_transfers_above_20k_24h = int((w_24h[ (w_24h["txn_type"].isin(["transfer","debit"])) & (w_24h["amount"]>=20000) ].shape[0]))

    count_outflow_1h = int(w_1h[ w_1h["txn_type"].isin(["transfer","debit"]) ].shape[0])
    count_outflow_3h = int(w_3h[ w_3h["txn_type"].isin(["transfer","debit"]) ].shape[0])
    count_outflow_6h = int(w_6h[ w_6h["txn_type"].isin(["transfer","debit"]) ].shape[0])

    # peak per hour in last 24h
    try:
        hourly = w_24h.set_index("timestamp").resample("1h")["txn_type"].apply(lambda s: (s=="transfer").sum())
        max_outflow_per_hour_24h = float(hourly.max()) if len(hourly)>0 else 0.0
    except Exception:
        max_outflow_per_hour_24h = 0.0

    # recipient entropy and unique recipients
    def entropy(s):
        s = s.dropna()
        if s.size == 0:
            return 0.0
        vc = s.value_counts(normalize=True)
        return - (vc * np.log2(vc)).sum()

    recipient_entropy_24h = entropy(w_24h[ w_24h["txn_type"].isin(["transfer","debit"]) ]["recipient_account_id"]) if "recipient_account_id" in hist.columns else 0.0
    recipient_entropy_7d = entropy(w_7d[ w_7d["txn_type"].isin(["transfer","debit"]) ]["recipient_account_id"]) if "recipient_account_id" in hist.columns else 0.0
    unique_recipients_24h = int(w_24h[ w_24h["txn_type"].isin(["transfer","debit"]) ]["recipient_account_id"].nunique()) if "recipient_account_id" in hist.columns else 0
    unique_recipients_7d = int(w_7d[ w_7d["txn_type"].isin(["transfer","debit"]) ]["recipient_account_id"].nunique()) if "recipient_account_id" in hist.columns else 0
    new_recipient_fraction_24h = unique_recipients_24h / (unique_recipients_7d + 1e-9)

    # last inbound amount and time since
    last_inflow = hist[ hist["txn_type"]=="credit" ]
    if len(last_inflow) > 0:
        last_inflow_ts = last_inflow["timestamp"].max()
        last_inflow_amount = float(last_inflow.loc[last_inflow["timestamp"]==last_inflow_ts, "amount"].iloc[-1])
        time_since_last_inflow_hours = (ts - last_inflow_ts).total_seconds()/3600.0
    else:
        last_inflow_amount = 0.0
        time_since_last_inflow_hours = 9999.0

    inflow_to_outflow_ratio_24h = last_inflow_amount / (sum_outflow_24h + 1e-9)
    is_large_inflow_recent = 1 if (last_inflow_amount >= 500000 and time_since_last_inflow_hours <= 48) else 0

    # build account_agg dict compatible with make_features
    account_agg = {
        "unique_recipients": unique_recipients,
        "outbound_transfer_count": outbound_transfer_count,
        "total_txn_count": total_txn_count,
        "avg_amount_cust": avg_amount_cust,
        "base_risk": base_risk,
        "account_age_days": account_age_days,
        "balance": balance,
        "txns_24h": float(w_24h.shape[0]),
        "txns_7d": float(w_7d.shape[0]),
        "last_inflow_amount": last_inflow_amount,
        "time_since_last_inflow_hours": time_since_last_inflow_hours,
        "sum_outflow_24h": sum_outflow_24h,
        "count_outflow_24h": count_outflow_24h,
        "median_outflow_24h": median_outflow_24h,
        "max_outflow_24h": max_outflow_24h,
        "inflow_to_outflow_ratio_24h": inflow_to_outflow_ratio_24h,
        "is_large_inflow_recent": is_large_inflow_recent,
        "num_transfers_above_20k_24h": num_transfers_above_20k_24h,
        # burst & recipient features
        "count_outflow_1h": count_outflow_1h,
        "count_outflow_3h": count_outflow_3h,
        "count_outflow_6h": count_outflow_6h,
        "max_outflow_per_hour_24h": max_outflow_per_hour_24h,
        "recipient_entropy_24h": recipient_entropy_24h,
        "recipient_entropy_7d": recipient_entropy_7d,
        "unique_recipients_24h": unique_recipients_24h,
        "unique_recipients_7d": unique_recipients_7d,
        "new_recipient_fraction_24h": new_recipient_fraction_24h
    }

    # Score using existing scorer (which uses persisted pipeline + iso)
    scores = score_transaction(txn, account_agg, alpha=alpha)
    if alert_threshold is not None:
        scores["alert"] = scores["fused_score"] >= alert_threshold

    scores["account_agg"] = account_agg
    return scores
