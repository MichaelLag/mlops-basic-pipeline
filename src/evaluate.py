from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def main() -> None:
    artifacts_dir = Path("artifacts")
    model_path = artifacts_dir / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run `python src/train.py` first."
        )

    # Load model
    model = joblib.load(model_path)

    # Load data (same dataset + same split settings as train.py)
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save evaluation outputs
    eval_out = {
        "accuracy": float(acc),
        "classification_report": report,
    }
    (artifacts_dir / "eval.json").write_text(json.dumps(eval_out, indent=2))

    print(f"Eval complete. Accuracy={acc:.4f}")
    print(f"Saved eval to {artifacts_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
