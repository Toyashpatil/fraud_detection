import os
import pickle
from sklearn.model_selection import train_test_split

from data_generator import generate_synthetic_data
from smurf_injector import inject_smurfing, inject_structuring
from feature_engineering import create_features
from model_training import train_supervised, train_unsupervised
from evaluate import evaluate_model

def main():
    print("Generating synthetic data...")
    df, customers = generate_synthetic_data()

    print("Injecting classic smurfing patterns...")
    df = inject_smurfing(df, customers, n_smurfers=30)

    print("Injecting structuring (large inflow -> many medium outflows)...")
    df = inject_structuring(df, customers,
                            n_structurers=12,
                            inflow_amount=1_000_000,
                            outflow_amount_range=(20000,30000),
                            n_outflows_range=(20,60),
                            window_hours=48)

    print("Building features...")
    df, X, y = create_features(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

    print("Training supervised model...")
    clf = train_supervised(X_train, y_train)

    print("Training unsupervised model...")
    iso = train_unsupervised(X_train)

    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    with open("models/rf_smurf_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("models/iso_smurf_model.pkl", "wb") as f:
        pickle.dump(iso, f)

    print("Evaluating and saving artifacts...")
    evaluate_model(clf, iso, X_test, y_test, outdir="data")

    # Save full synthetic dataset for inspection
    df.to_csv("data/synthetic_smurf_transactions.csv", index=False)
    print("Saved synthetic dataset to data/synthetic_smurf_transactions.csv")
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
