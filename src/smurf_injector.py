import numpy as np

def inject_smurfing(df, customers, n_smurfers=30):
    smurfers = np.random.choice(customers.customer_id, n_smurfers, replace=False)

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

        chosen = cand_transfer.sample(frac=0.75)
        df.loc[chosen.index, "amount"] = df.loc[chosen.index, "amount"].apply(lambda x: round(max(5, x*0.05),2))
        df.loc[chosen.index, "recipient_account_id"] = [f"R_{s}_{i}" for i in range(len(chosen))]
        df.loc[chosen.index, "label"] = 1

    return df
