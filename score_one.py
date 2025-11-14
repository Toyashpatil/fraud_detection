from datetime import datetime
from src.infer import score_transaction

txn = {"amount": 25000.0, "timestamp": datetime.utcnow(), "txn_type": "transfer"}
account_agg = {
    "unique_recipients": 40,
    "outbound_transfer_count": 45,
    "total_txn_count": 200,
    "avg_amount_cust": 1500,
    "base_risk": 0.3,
    "account_age_days": 200,
    "balance": 200000,
    "txns_24h": 50,
    "txns_7d": 200,
    # new structuring fields
    "last_inflow_amount": 1000000.0,
    "time_since_last_inflow_hours": 2.0,
    "sum_outflow_24h": 900000.0,
    "count_outflow_24h": 36,
    "median_outflow_24h": 25000.0,
    "max_outflow_24h": 30000.0,
    "inflow_to_outflow_ratio_24h": 1.111,   # last_inflow / sum_outflow
    "is_large_inflow_recent": 1,
    "num_transfers_above_20k_24h": 36
}

print(score_transaction(txn, account_agg))
