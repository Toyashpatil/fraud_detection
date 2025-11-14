from sklearn.ensemble import RandomForestClassifier, IsolationForest

def train_supervised(X_train, y_train, n_estimators=200):
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def train_unsupervised(X_train, n_estimators=200, contamination=0.02):
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42
    )
    iso.fit(X_train)
    return iso
