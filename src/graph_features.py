import pandas as pd
import networkx as nx


def build_bipartite_graph(df, recent_window_days=7):
    """
    Build a lightweight bipartite graph (customer -> recipient) using transactions from the last `recent_window_days`.
    Returns a NetworkX graph with bipartite node attribute set to 'customer' or 'recipient'.
    """
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = df["timestamp"].max() - pd.Timedelta(days=recent_window_days)
        df = df[df["timestamp"] >= cutoff]

    B = nx.Graph()
    # add nodes and edges for transfers only
    if "txn_type" in df.columns:
        df_edges = df[df["txn_type"].isin(["transfer","debit"])].dropna(subset=["recipient_account_id"])
    else:
        df_edges = df.dropna(subset=["recipient_account_id"])

    for _, row in df_edges.iterrows():
        cust = f"C_{row['customer_id']}"
        rec = f"R_{row['recipient_account_id']}"
        if not B.has_node(cust):
            B.add_node(cust, bipartite="customer")
        if not B.has_node(rec):
            B.add_node(rec, bipartite="recipient")
        if B.has_edge(cust, rec):
            B[cust][rec]["weight"] += 1
        else:
            B.add_edge(cust, rec, weight=1)

    return B


def compute_graph_features(df, recent_window_days=7, top_k_shared=5):
    """
    Compute per-customer graph features from transaction DataFrame.
    Returns a DataFrame indexed by `customer_id` with features:
      - customer_degree: number of unique recipients
      - avg_recipient_degree: average degree of connected recipients
      - top_shared_recipient_count: number of recipients shared with other customers more than once
      - component_size: size of connected component containing the customer
    """
    B = build_bipartite_graph(df, recent_window_days=recent_window_days)
    # extract customer nodes
    customers = [n for n, d in B.nodes(data=True) if d.get("bipartite") == "customer"]
    rows = []
    # precompute recipient degrees
    recip_degrees = {n: d for n, d in B.degree() if B.nodes[n].get("bipartite") == "recipient"}

    for cust in customers:
        customer_id = cust.replace("C_", "")
        degree = B.degree(cust)
        # recipients connected to this customer
        recs = [n for n in B.neighbors(cust) if B.nodes[n].get("bipartite") == "recipient"]
        avg_rec_deg = float(sum(B.degree(r) for r in recs) / (len(recs) or 1))

        # shared recipient count: recipients with degree >1
        shared_count = sum(1 for r in recs if B.degree(r) > 1)

        # component size
        try:
            comp = nx.node_connected_component(B, cust)
            comp_size = len([n for n in comp if B.nodes[n].get("bipartite") == "customer"])
        except Exception:
            comp_size = 1

        rows.append({
            "customer_id": customer_id,
            "customer_degree": int(degree),
            "avg_recipient_degree": float(avg_rec_deg),
            "shared_recipient_count": int(shared_count),
            "component_size": int(comp_size)
        })

    return pd.DataFrame(rows).set_index("customer_id")
