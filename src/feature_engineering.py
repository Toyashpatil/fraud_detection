import numpy as np
import pandas as pd

def create_features(df):
    """
    Input: df with columns including:
      - txn_id, customer_id, amount, txn_type, timestamp, weekday, hour, txns_24h, txns_7d, balance, base_risk, account_age_days, recipient_account_id
    Output:
      - df (with new features), X (features DataFrame), y (labels Series)
    """

    # Ensure timestamp is datetime
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Keep original transaction-level features
    df["amount_log"] = np.log1p(df["amount"])
    df["balance_to_amount"] = df["balance"] / (df["amount"] + 1)
    df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    # Flags for inflow/outflow
    df["is_outflow"] = df["txn_type"].isin(["transfer","debit"]).astype(int)
    df["is_inflow"] = df["txn_type"].isin(["credit"]).astype(int)

    # --- account-level aggregated features (existing) ---
    grouped = df.groupby("customer_id")
    df = df.merge(grouped["recipient_account_id"].nunique().rename("unique_recipients"), on="customer_id", how="left")
    df = df.merge(grouped.apply(lambda g: (g.txn_type=="transfer").sum()).rename("outbound_transfer_count"), on="customer_id", how="left")
    df = df.merge(grouped.size().rename("total_txn_count"), on="customer_id", how="left")
    df = df.merge(grouped["amount"].mean().rename("avg_amount_cust"), on="customer_id", how="left")

    df["recipient_rate"] = df["outbound_transfer_count"] / (df["total_txn_count"] + 1e-9)
    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount_cust"] + 1e-9)
    df["unique_recipients_per_transfer"] = df["unique_recipients"] / (df["outbound_transfer_count"] + 1e-9)
    df["smurf_rule_flag"] = ((df["unique_recipients"] > 10) & (df["outbound_transfer_count"] > 15) & (df["amount"] < 200)).astype(int)

    # --- New: rolling/time-window features for structuring detection ---
    # We'll compute per-customer time-window rolling stats for outflows and track last inbound info.
    # For efficiency, do groupby + apply with time-based rolling using timestamp as index.

    def compute_time_windows(g):
        g = g.sort_values("timestamp").set_index("timestamp")
        # rolling sums/counts of outflow in 24h
        outflow_amount = g["amount"].where(g["is_outflow"]==1, other=0.0)
        outflow_count = g["is_outflow"]
        g["sum_outflow_24h"] = outflow_amount.rolling("24h").sum().fillna(0)
        g["count_outflow_24h"] = outflow_count.rolling("24h").sum().fillna(0)

        # rolling median and max of outflow amounts in 24h
        g["median_outflow_24h"] = outflow_amount.replace(0, np.nan).rolling("24h").median().fillna(0)
        g["max_outflow_24h"] = outflow_amount.rolling("24h").max().fillna(0)

        # number of transfers >= 20k in last 24h
        high_transfer = g["amount"].where((g["is_outflow"]==1) & (g["amount"] >= 20000), other=0.0)
        # count non-zero entries in rolling window
        g["num_transfers_above_20k_24h"] = (high_transfer > 0).rolling("24h").sum().fillna(0)

        # last inbound (most recent prior credit amount) and timestamp
        # forward-fill last seen inbound amount and its timestamp
        inbound_amount = g["amount"].where(g["is_inflow"]==1)
        g["last_inflow_amount"] = inbound_amount.ffill().fillna(0)
        # create last inbound timestamp by carrying forward index where is_inflow
        inbound_ts = pd.Series(g.index.where(g["is_inflow"]==1, pd.NaT), index=g.index).ffill()
        g["last_inflow_ts"] = inbound_ts

        # time since last inbound in hours
        g["time_since_last_inflow_hours"] = (g.index.to_series() - g["last_inflow_ts"]).dt.total_seconds().div(3600).fillna(9999)

        # inflow to outflow ratio 24h (last_inflow_amount / sum_outflow_24h)
        g["inflow_to_outflow_ratio_24h"] = g["last_inflow_amount"] / (g["sum_outflow_24h"] + 1e-9)

        # flag for recent large inflow and transfers after it
        g["is_large_inflow_recent"] = ((g["last_inflow_amount"] >= 500000) & (g["time_since_last_inflow_hours"] <= 48)).astype(int)

        return g

    # Apply per-customer
    df = df.groupby("customer_id", group_keys=False).apply(compute_time_windows).reset_index(drop=False)

    # Fill NA for numeric columns
    df[[
        "sum_outflow_24h","count_outflow_24h","median_outflow_24h","max_outflow_24h",
        "num_transfers_above_20k_24h","last_inflow_amount","time_since_last_inflow_hours",
        "inflow_to_outflow_ratio_24h","is_large_inflow_recent"
    ]] = df[[
        "sum_outflow_24h","count_outflow_24h","median_outflow_24h","max_outflow_24h",
        "num_transfers_above_20k_24h","last_inflow_amount","time_since_last_inflow_hours",
        "inflow_to_outflow_ratio_24h","is_large_inflow_recent"
    ]].fillna(0)

    # --- Final feature columns (keep old ones + new ones) ---
    feature_cols = [
        # original set
        "amount","amount_log","balance","balance_to_amount","txns_24h","txns_7d",
        "is_weekend","hour_sin","hour_cos","unique_recipients","outbound_transfer_count",
        "recipient_rate","avg_amount_cust","amount_to_avg_ratio","unique_recipients_per_transfer",
        "base_risk","smurf_rule_flag","account_age_days",
        # new structuring features
        "last_inflow_amount","time_since_last_inflow_hours","sum_outflow_24h","count_outflow_24h",
        "median_outflow_24h","max_outflow_24h","inflow_to_outflow_ratio_24h","is_large_inflow_recent",
        "num_transfers_above_20k_24h"
    ]

    # Ensure all requested columns exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    X = df[feature_cols].fillna(0)
    y = df["label"].astype(int)
    return df, X, y
