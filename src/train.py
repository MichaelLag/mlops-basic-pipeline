# src/train.py
import json
import os
import platform
import sys
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import load_config  # note: same folder import

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

def main() -> None:
    cfg = load_config()

    test_size = cfg["data"]["test_size"]
    random_state = cfg["data"]["random_state"]
    model_params = cfg["model"]["params"]

    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(
        random_state=random_state,
        **model_params,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    joblib.dump(model, ARTIFACTS_DIR / "model.joblib")

    metrics = {
        "accuracy": acc,
        "config": cfg,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Training accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
