from datetime import datetime
from src.infer import score_transaction

account_agg = {
    "unique_recipients": 200,
    "outbound_transfer_count": 300,
    "total_txn_count": 320,
    "avg_amount_cust": 1500,
    "base_risk": 0.8,
    "account_age_days": 120,
    "balance": 5000,
    "txns_24h": 120,
    "txns_7d": 300
}
txn = {"amount": 12.0, "timestamp": datetime.utcnow(), "txn_type":"transfer"}

print(score_transaction(txn, account_agg))
# returns supervised_proba, iso_score, iso_norm, fused_score
