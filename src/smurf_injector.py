import numpy as np
import pandas as pd

def inject_smurfing(df, customers, n_smurfers=30, seed=42):
    """
    Classic smurfing: many small transfers from one origin to many unique recipients.
    """
    np.random.seed(seed)
    smurfers = np.random.choice(customers.customer_id, n_smurfers, replace=False)

    case_id_ctr = 0
    for s in smurfers:
        mask = df.customer_id == s
        cand = df[mask]

        if cand.shape[0] < 40:
            idxs = cand.sample(min(40, cand.shape[0])).index
            df.loc[idxs, "txn_type"] = "transfer"

        cand_transfer = df[mask & (df.txn_type=="transfer")]
        if cand_transfer.shape[0] < 20:
            idxs = df[mask].sample(20, replace=True).index
            cand_transfer = df.loc[idxs]

        chosen = cand_transfer.sample(frac=0.75, replace=False)
        df.loc[chosen.index, "amount"] = df.loc[chosen.index, "amount"].apply(lambda x: round(max(5, x*0.05),2))
        df.loc[chosen.index, "recipient_account_id"] = [f"R_{s}_{i}" for i in range(len(chosen))]
        # assign a fraud_case_id per smurfing account
        fraud_case_id = f"smurf_{s}_{seed}_{case_id_ctr}"
        df.loc[chosen.index, "label"] = 1
        df.loc[chosen.index, "fraud_case_id"] = fraud_case_id
        # mark account-level label for all transactions of this customer
        df.loc[mask, "account_label"] = 1
        case_id_ctr += 1

    return df

def inject_structuring(df, customers, n_structurers=10, inflow_amount=1_000_000,
                       outflow_amount_range=(20000,30000), n_outflows_range=(20,60),
                       window_hours=48, seed=42):
    """
    Structuring/Layering injection:
    For selected accounts, add a large inbound (credit) and then many medium outflows within a window.
    This function will append new rows to df to simulate these events and label the outflows as fraud.
    """
    np.random.seed(seed)
    # Work on a copy and append
    new_rows = []
    next_txn_id = df["txn_id"].max() + 1

    structurers = np.random.choice(customers.customer_id, n_structurers, replace=False)
    for s in structurers:
        # pick a random timestamp from that customer's existing transactions as base time
        cust_txns = df[df.customer_id==s]
        if cust_txns.empty:
            continue
        base_time = cust_txns["timestamp"].sample(1).iloc[0]

        # 1) append a large inbound credit (inflow)
        inflow_ts = base_time + pd.Timedelta(hours=np.random.uniform(0.1, 24))
        new_rows.append({
            "txn_id": next_txn_id,
            "customer_id": s,
            "amount": float(inflow_amount),
            "txn_type": "credit",
            "origin_country": "IN",
            "dest_country": "IN",
            "merchant_category": "remittance",
            "timestamp": inflow_ts,
            "weekday": inflow_ts.weekday(),
            "hour": inflow_ts.hour,
            "txns_24h": 0,
            "txns_7d": 0,
            "balance": float(inflow_amount),  # approximate
            "recipient_account_id": None,
            "label": 0  # inbound itself not labeled as fraud, outflows will be
        })
        next_txn_id += 1

        # 2) schedule many medium outflows within next window_hours
        n_out = np.random.randint(n_outflows_range[0], n_outflows_range[1]+1)
        for i in range(n_out):
            out_ts = inflow_ts + pd.Timedelta(hours=np.random.uniform(0.01, window_hours))
            amount = float(np.round(np.random.uniform(outflow_amount_range[0], outflow_amount_range[1]),2))
            new_rows.append({
                "txn_id": next_txn_id,
                "customer_id": s,
                "amount": amount,
                "txn_type": "transfer",
                "origin_country": "IN",
                "dest_country": "IN",
                "merchant_category": "remittance",
                "timestamp": out_ts,
                "weekday": out_ts.weekday(),
                "hour": out_ts.hour,
                "txns_24h": 0,
                "txns_7d": 0,
                "balance": float(max(0, inflow_amount - amount*(i+1))),
                "recipient_account_id": f"R_struct_{s}_{i}",
                "label": 1  # mark these outflows as suspicious/fraud
            })
            next_txn_id += 1

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # For structuring cases, assign fraud_case_id to the outflows per account
        # Find the groups of rows per customer and assign case ids
        struct_case_ctr = 0
        for s in structurers:
            mask_new = df_new["customer_id"] == s
            if not mask_new.any():
                continue
            # outflows for this structurer are labeled as fraud (label==1)
            fraud_mask = mask_new & (df_new["label"] == 1)
            if fraud_mask.any():
                cid = f"struct_{s}_{seed}_{struct_case_ctr}"
                df_new.loc[fraud_mask, "fraud_case_id"] = cid
                # mark account_label on the original df for that customer too
                df.loc[df["customer_id"]==s, "account_label"] = 1
                struct_case_ctr += 1
        # Ensure types align
        df = pd.concat([df, df_new], ignore_index=True, sort=False)
        # sort by timestamp to maintain chronological order
        df = df.sort_values(["customer_id","timestamp"]).reset_index(drop=True)

    return df
