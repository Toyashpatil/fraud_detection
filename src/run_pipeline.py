import os
import pandas as pd
from sklearn.model_selection import train_test_split

from data_generator import generate_synthetic_data
from smurf_injector import inject_smurfing
from feature_engineering import create_features
from model_training import train_supervised, train_unsupervised
from evaluate import evaluate_model

def main():
    print("Generating synthetic data...")
    df, customers = generate_synthetic_data()

    print("Injecting smurfing...")
    df = inject_smurfing(df, customers)

    print("Engineering features...")
    df, X, y = create_features(df)

    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    print("Training supervised model...")
    clf = train_supervised(X_train, y_train)

    print("Training unsupervised model...")
    iso = train_unsupervised(X_train)

    os.makedirs("data/", exist_ok=True)
    os.makedirs("models/", exist_ok=True)

    print("Saving models...")
    import pickle
    pickle.dump(clf, open("models/rf_smurf_model.pkl","wb"))
    pickle.dump(iso, open("models/iso_smurf_model.pkl","wb"))

    print("Evaluating...")
    evaluate_model(clf, iso, X_test, y_test, outdir="data/")

    print("DONE.")

if __name__ == "__main__":
    main()
