from datetime import datetime, timedelta
import pandas as pd
from src.infer import score_stream

now = datetime.utcnow()
hist = pd.DataFrame([
    {"timestamp": now - timedelta(hours=50), "amount": 1000000, "txn_type": "credit", "recipient_account_id": None, "balance": 500000},
    {"timestamp": now - timedelta(hours=47), "amount": 25000, "txn_type": "transfer", "recipient_account_id": "R1"},
    {"timestamp": now - timedelta(hours=46), "amount": 24000, "txn_type": "transfer", "recipient_account_id": "R2"},
    {"timestamp": now - timedelta(hours=2), "amount": 26000, "txn_type": "transfer", "recipient_account_id": "R3"},
    {"timestamp": now - timedelta(minutes=30), "amount": 150, "txn_type": "transfer", "recipient_account_id": "R4"},
    {"timestamp": now - timedelta(minutes=20), "amount": 120, "txn_type": "transfer", "recipient_account_id": "R5"},
])

txn = {"timestamp": now, "amount": 25000, "txn_type": "transfer", "recipient_account_id": "R6"}
res = score_stream(txn, hist, alpha=0.7, alert_threshold=0.5)
print(res)
