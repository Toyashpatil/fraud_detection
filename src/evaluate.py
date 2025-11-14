import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score

def evaluate_model(clf, iso, X_test, y_test, outdir="data"):
    os.makedirs(outdir, exist_ok=True)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    print("\n### SUPERVISED MODEL REPORT ###")
    print(classification_report(y_test, y_pred, digits=4))

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except:
        roc = float("nan")
    print(f"ROC AUC = {roc:.4f}")
    print(f"PR AUC = {pr_auc:.4f}")

    iso_scores = -iso.decision_function(X_test)
    k = int(y_test.sum()) if y_test.sum()>0 else 10
    topk = iso_scores.argsort()[-k:]
    detected = int(y_test.iloc[topk].sum()) if k>0 else 0
    print(f"IsolationForest Precision@K = {detected}/{k} = {detected/k if k>0 else 0:.4f}")

    # Save PR curve
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.title("PR Curve (Smurf Detection)")
    pr_path = os.path.join(outdir, "pr_curve.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()
    print("Saved PR curve to:", pr_path)

    # Save top alerts (include additional window features for review)
    alerts_df = pd.DataFrame(X_test).reset_index(drop=True)
    alerts_df["proba"] = y_proba
    alerts_df["label"] = y_test.reset_index(drop=True)
    alerts_df = alerts_df.sort_values("proba", ascending=False).head(500)
    alerts_path = os.path.join(outdir, "top_alerts.csv")
    alerts_df.to_csv(alerts_path, index=False)
    print("Saved top alerts to:", alerts_path)
