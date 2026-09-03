"""Pipeline orchestrator. Runs every layer end to end, in order, by
importing and calling each scripts/*.py file's own main() -- no pipeline
logic is duplicated here, this file is purely sequencing. The single
source of truth for each step stays in its scripts/*.py file; edit that
file, not this one, to change what a step does.

Order: build_folds -> tune_svdpp -> train_svdpp -> train_decaypop ->
generate_hybrid_predictions -> compute_metrics -> build_segments ->
user_sensitivity.

Note: scripts/2a. train_svdpp.py skips folds that already have a
model.pkl + top20_unseen.csv on disk. If you retuned hyperparameters
(step 1a) and want a real full retrain rather than a skip, delete
results/svdpp/ first.

Run from project root:
    python index.py
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

STEPS = [
    "1. build_folds.py",
    "1a. tune_svdpp.py",
    "2a. train_svdpp.py",
    "2b. train_decaypop.py",
    "3. generate_hybrid_predictions.py",
    "4. compute_metrics.py",
    "5. build_segments.py",
    "6. user_sensitivity.py",
]


def _load_step_module(script_name: str) -> ModuleType:
    path = SCRIPTS_DIR / script_name
    module_name = "pipeline_step_" + path.stem.replace(" ", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    started = time.time()
    for step_name in STEPS:
        print(f"\n{'=' * 70}\n>>> {step_name}\n{'=' * 70}")
        step_started = time.time()
        module = _load_step_module(step_name)
        module.main()
        print(f"--- {step_name} selesai ({time.time() - step_started:.1f}s) ---")

    print(f"\nPipeline selesai penuh dalam {time.time() - started:.1f}s.")


if __name__ == "__main__":
    main()
