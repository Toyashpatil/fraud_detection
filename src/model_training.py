from sklearn.ensemble import RandomForestClassifier, IsolationForest

def train_supervised(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    return clf

def train_unsupervised(X_train):
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42
    )
    iso.fit(X_train)
    return iso
