from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_model(clf, iso, X_test, y_test, outdir="data/"):

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]

    print("\n### SUPERVISED MODEL REPORT ###")
    print(classification_report(y_test, y_pred))

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    roc = roc_auc_score(y_test, y_proba)

    print(f"ROC AUC = {roc}")
    print(f"PR AUC = {pr_auc}")

    iso_scores = -iso.decision_function(X_test)
    k = int(y_test.sum())
    topk = np.argsort(iso_scores)[-k:]
    detected = y_test.iloc[topk].sum()

    print(f"IsolationForest Precision@K = {detected/k}")

    # Save PR curve
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid()
    plt.title("PR Curve (Smurf Detection)")
    plt.savefig(os.path.join(outdir, "pr_curve.png"))
    plt.close()

    # Alerts CSV
    alerts = pd.DataFrame({
        "proba": y_proba,
        "label": y_test
    }).sort_values("proba", ascending=False).head(500)

    alerts.to_csv(os.path.join(outdir, "top_alerts.csv"), index=False)
