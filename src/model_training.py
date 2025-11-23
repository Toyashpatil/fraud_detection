from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os


def build_training_pipeline(n_estimators=200, use_scaler=True):
    """Return an sklearn Pipeline that contains optional scaler + RandomForestClassifier."""
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("rf", RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )))
    return Pipeline(steps)


def train_supervised_pipeline(X_train, y_train, n_estimators=200, use_scaler=True):
    """Build, fit and return a pipeline containing scaler (optional) and RF classifier."""
    pipeline = build_training_pipeline(n_estimators=n_estimators, use_scaler=use_scaler)
    pipeline.fit(X_train, y_train)
    return pipeline


def train_unsupervised(X_train, n_estimators=200, contamination=0.02):
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42
    )
    iso.fit(X_train)
    return iso


def persist_artifacts(pipeline, iso_model, feature_list, outdir="models"):
    """Persist pipeline, isolation model and feature list to disk under `outdir`."""
    os.makedirs(outdir, exist_ok=True)
    pipeline_path = os.path.join(outdir, "feature_pipeline.pkl")
    iso_path = os.path.join(outdir, "iso_smurf_model.pkl")
    feature_list_path = os.path.join(outdir, "feature_list.json")

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(iso_model, iso_path)
    with open(feature_list_path, "w") as f:
        json.dump(feature_list, f)

    return pipeline_path, iso_path, feature_list_path
