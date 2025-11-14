import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

def generate_synthetic_data(n_customers=2500, n_txns=60000):
    np.random.seed(42)
    random.seed(42)

    start = datetime(2024,1,1)
    end = datetime(2025,11,1)

    customers = pd.DataFrame({
        "customer_id": np.arange(1, n_customers+1),
        "account_age_days": np.random.randint(10, 4000, n_customers),
        "base_risk": np.random.beta(2, 18, n_customers)
    })

    merchant_categories = ["retail","grocery","utilities","travel","salary","remittance","services","others"]
    txn_types = ["debit","credit","transfer","cash_withdrawal"]

    rows = []
    for i in range(n_txns):
        cust = customers.sample(1).iloc[0]
        cid = int(cust.customer_id)
        amount = float(np.round(np.random.exponential(scale=120), 2))

        if np.random.rand() < 0.015:
            amount *= np.random.randint(5,60)

        txn_type = np.random.choice(txn_types, p=[0.45,0.2,0.25,0.1])
        origin_country = "IN"
        dest_country = "IN"
        merchant = np.random.choice(merchant_categories)
        ts = random_date(start,end)

        rows.append([
            i+1, cid, amount, txn_type,
            origin_country, dest_country, merchant,
            ts, ts.weekday(), ts.hour,
            np.random.poisson(0.8),
            np.random.poisson(2.0),
            float(max(0, np.round(np.random.normal(5000,2000),2)))
        ])

    df = pd.DataFrame(rows, columns=[
        "txn_id","customer_id","amount","txn_type","origin_country","dest_country",
        "merchant_category","timestamp","weekday","hour","txns_24h","txns_7d",
        "balance"
    ])

    df = df.merge(customers, on="customer_id", how="left")
    df["recipient_account_id"] = None
    df["label"] = 0

    return df, customers
