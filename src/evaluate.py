# src/evaluate.py
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import load_config

ARTIFACTS_DIR = Path("artifacts")

def main() -> None:
    cfg = load_config()
    test_size = cfg["data"]["test_size"]
    random_state = cfg["data"]["random_state"]

    model = joblib.load(ARTIFACTS_DIR / "model.joblib")

    X, y = load_iris(return_X_y=True)
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    eval_metrics = {"accuracy": acc, "config": cfg}

    with open(ARTIFACTS_DIR / "eval.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Evaluation accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
