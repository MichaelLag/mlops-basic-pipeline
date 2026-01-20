# src/train.py
import json
import os
import sys
import platform
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

def main() -> None:
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=200,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")

    metrics = {
        "accuracy": acc,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "model_params": model.get_params(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
