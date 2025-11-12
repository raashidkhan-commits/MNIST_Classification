from functools import lru_cache
from pathlib import Path
import joblib

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "sgd_clf.pkl"

@lru_cache(maxsize=1)
def load_data():
    """Load MNIST CSVs once per process and cache in-memory."""
    import pandas as pd
    X = pd.read_csv(DATA_DIR / "mnist_data.csv").to_numpy()
    y = pd.read_csv(DATA_DIR / "mnist_target.csv").to_numpy().ravel()
    return X, y

def load_split():
    X, y = load_data()
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[60000:], y[60000:]
    return X_train, y_train, X_test, y_test

def train_and_save(X_train, y_train, path=MODEL_PATH, random_state=42):
    """Train SGD on provided data and save to disk."""
    from sklearn.linear_model import SGDClassifier
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    joblib.dump(clf, path)
    return clf

def load_model(path=MODEL_PATH):
    """Load a saved model from disk (raises if not found)."""
    return joblib.load(path)

def get_or_train_model(force_train=False):
    """
    Return a trained model. If a saved model exists and force_train is False,
    load it; otherwise train on the training split and save the model.
    """
    if (not force_train) and MODEL_PATH.exists():
        return load_model(MODEL_PATH)
    X_train, y_train, X_test, y_test = load_split()
    y_train_5 = y_train == 5
    return train_and_save(X_train, y_train_5, path=MODEL_PATH)