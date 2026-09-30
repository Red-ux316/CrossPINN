"""Smoke test: a 5-epoch debug training of the C-PINN model on Problem 1 runs end to end on CPU."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_debug_training_runs():
    result = subprocess.run(
        [sys.executable, "tools/train.py", "--prob_id", "1", "--model", "tlp", "--debug", "--no_wandb"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "Starting training" in result.stdout
