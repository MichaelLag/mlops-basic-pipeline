# src/evaluate.py
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ARTIFACTS_DIR = Path("artifacts")

RANDOM_STATE = 42
TEST_SIZE = 0.2

def main():
    model = joblib.load(ARTIFACTS_DIR / "model.joblib")

    X, y = load_iris(return_X_y=True)
    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    eval_metrics = {
        "accuracy": acc,
    }

    with open(ARTIFACTS_DIR / "eval.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"Evaluation accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
