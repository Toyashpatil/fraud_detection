import numpy as np
import pandas as pd

def create_features(df):
    df["amount_log"] = np.log1p(df["amount"])
    df["balance_to_amount"] = df["balance"] / (df["amount"] + 1)
    df["is_weekend"] = df["weekday"].isin([5,6]).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    grouped = df.groupby("customer_id")
    df = df.merge(grouped["recipient_account_id"].nunique().rename("unique_recipients"), on="customer_id")
    df = df.merge(grouped.apply(lambda g: (g.txn_type=="transfer").sum()).rename("outbound_transfer_count"), on="customer_id")
    df = df.merge(grouped.size().rename("total_txn_count"), on="customer_id")
    df = df.merge(grouped["amount"].mean().rename("avg_amount_cust"), on="customer_id")

    df["recipient_rate"] = df["outbound_transfer_count"] / (df["total_txn_count"] + 1e-9)
    df["amount_to_avg_ratio"] = df["amount"] / (df["avg_amount_cust"] + 1e-9)
    df["unique_recipients_per_transfer"] = df["unique_recipients"] / (df["outbound_transfer_count"] + 1e-9)

    df["smurf_rule_flag"] = (
        (df["unique_recipients"] > 10) &
        (df["outbound_transfer_count"] > 15) &
        (df["amount"] < 200)
    ).astype(int)

    feature_cols = [
        "amount","amount_log","balance","balance_to_amount","txns_24h","txns_7d",
        "is_weekend","hour_sin","hour_cos","unique_recipients","outbound_transfer_count",
        "recipient_rate","avg_amount_cust","amount_to_avg_ratio","unique_recipients_per_transfer",
        "base_risk","smurf_rule_flag","account_age_days"
    ]

    return df, df[feature_cols], df["label"]
