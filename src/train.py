from __future__ import annotations

import json
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


def main() -> None:
    # --- Data (built-in dataset to keep this project self-contained) ---
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Model ---
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # --- Evaluation (quick sanity metric) ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # --- Save artifacts ---
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    joblib.dump(model, artifacts_dir / "model.joblib")

    metrics = {"accuracy": float(acc)}
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Train complete. Accuracy={acc:.4f}")
    print(f"Saved model to {artifacts_dir / 'model.joblib'}")
    print(f"Saved metrics to {artifacts_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()