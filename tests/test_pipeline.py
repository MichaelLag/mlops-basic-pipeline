import json
import subprocess
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")


def run_train_and_eval():
    # Run scripts as subprocesses to mimic CI behavior closely
    subprocess.run(["python", "src/train.py"], check=True)
    subprocess.run(["python", "src/evaluate.py"], check=True)


def test_artifacts_created():
    run_train_and_eval()
    assert (ARTIFACTS_DIR / "model.joblib").exists()
    assert (ARTIFACTS_DIR / "metrics.json").exists()
    assert (ARTIFACTS_DIR / "eval.json").exists()


def test_metrics_schema():
    run_train_and_eval()
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
    assert "accuracy" in metrics
    assert "config" in metrics
    assert "git_sha" in metrics


def test_eval_schema():
    run_train_and_eval()
    ev = json.loads((ARTIFACTS_DIR / "eval.json").read_text())
    assert "accuracy" in ev
    assert 0.0 <= ev["accuracy"] <= 1.0
