"""Layer 1a entrypoint. Optuna search for SVD++ hyperparameters (d, alpha,
lambda, decay).

Ported from recsys-project/src/svdpp.py::tune_hyperparams (search space,
TPE sampler, MedianPruner, fast-CV-with-early-stopping objective) -- the
one part of the old codebase that did hyperparameter search at all.
Adapted for this pipeline's leak-safe split: the old function derived its
own ad-hoc n-fold split of raw ratings just for tuning speed, independent
of (and inconsistent with) the 10-outer-fold CV used for final training.
Here, tuning instead reuses the first SVDPP_TUNE_N_FOLDS outer folds'
own train_inner_norm.csv/val_inner_norm.csv (already written by
build_folds.py) -- so a trial's score is measured exactly the way
train_svdpp.py's early stopping measures it, and Test is never touched
during tuning either.

Writes:
    results/svdpp_tuning/best_hyperparams.json
    results/svdpp_tuning/trials.csv

Run from project root (after scripts/build_folds.py):
    python "scripts/1a. tune_svdpp.py"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    SVDPP_TUNE_N_TRIALS,
    SVDPP_TUNE_N_FOLDS,
    SVDPP_TUNE_N_EPOCHS,
    SVDPP_TUNE_PATIENCE,
    SVDPP_TUNE_SEARCH_SPACE,
    SVDPP_INIT_SEED,
)
from src.svdpp import init_params, build_implicit_feedback, train_one_epoch, compute_rmse_mae

TUNE_OUT = RESULTS_DIR / "svdpp_tuning"


def _load_fold_ratings(fold_id: int):
    fold_dir = PROCESSED_DIR / f"fold{fold_id}"
    train_inner = pd.read_csv(fold_dir / "train_inner_norm.csv")
    val_inner = pd.read_csv(fold_dir / "val_inner_norm.csv")
    train_ratings = list(zip(train_inner["user_id"], train_inner["movie_id"], train_inner["rating"]))
    val_ratings = list(zip(val_inner["user_id"], val_inner["movie_id"], val_inner["rating"]))
    mu = float(train_inner["rating"].mean())
    return train_ratings, val_ratings, mu


def main() -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with open(PROCESSED_DIR / "manifest.json") as f:
        manifest = json.load(f)
    n_users_max_id = manifest["max_user_id"]
    n_items_max_id = manifest["max_movie_id"]

    tuning_folds = list(range(SVDPP_TUNE_N_FOLDS))
    fold_data = {fold_id: _load_fold_ratings(fold_id) for fold_id in tuning_folds}
    print(f"Tuning pakai {len(tuning_folds)} fold (train_inner/val_inner): {tuning_folds}")

    def objective(trial: "optuna.Trial") -> float:
        d = trial.suggest_categorical("d", SVDPP_TUNE_SEARCH_SPACE["d"])
        alpha_lo, alpha_hi = SVDPP_TUNE_SEARCH_SPACE["alpha"]
        lam_lo, lam_hi = SVDPP_TUNE_SEARCH_SPACE["lambda"]
        decay_lo, decay_hi = SVDPP_TUNE_SEARCH_SPACE["decay"]
        alpha = trial.suggest_float("alpha", alpha_lo, alpha_hi, log=True)
        lam = trial.suggest_float("lambda", lam_lo, lam_hi, log=True)
        decay = trial.suggest_float("decay", decay_lo, decay_hi)
        hp = {"d": d, "alpha": alpha, "lambda": lam, "decay": decay}

        fold_rmses = []
        for fold_id in tuning_folds:
            train_ratings, val_ratings, mu = fold_data[fold_id]
            impl_N, impl_inv = build_implicit_feedback(train_ratings)
            params = init_params(mu, n_users_max_id, n_items_max_id, d=d, seed=SVDPP_INIT_SEED)
            hp_f = dict(hp)
            rng = np.random.default_rng(SVDPP_INIT_SEED + fold_id)

            best_rmse_f = float("inf")
            no_improve = 0
            for _epoch in range(1, SVDPP_TUNE_N_EPOCHS + 1):
                params, _ = train_one_epoch(train_ratings, params, hp_f, impl_N, impl_inv, rng)
                rmse_f, _ = compute_rmse_mae(val_ratings, params, impl_N, impl_inv)
                hp_f["alpha"] *= hp_f["decay"]
                if rmse_f < best_rmse_f:
                    best_rmse_f = rmse_f
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= SVDPP_TUNE_PATIENCE:
                        break

            fold_rmses.append(best_rmse_f)
            trial.report(float(np.mean(fold_rmses)), fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_rmses))

    def _progress(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        val = f"{trial.value:.4f}" if trial.value is not None else "pruned"
        print(f"Trial {trial.number + 1:>2}/{SVDPP_TUNE_N_TRIALS} | RMSE={val} | Best={study.best_value:.4f} | {trial.params}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SVDPP_INIT_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )
    study.optimize(objective, n_trials=SVDPP_TUNE_N_TRIALS, callbacks=[_progress])

    best_hp = study.best_params
    TUNE_OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "d": best_hp["d"],
        "alpha": best_hp["alpha"],
        "lambda": best_hp["lambda"],
        "decay": best_hp["decay"],
        "best_val_rmse_mean": study.best_value,
        "n_trials": len(study.trials),
        "tuning_folds": tuning_folds,
    }
    with open(TUNE_OUT / "best_hyperparams.json", "w") as f:
        json.dump(result, f, indent=2)
    study.trials_dataframe().to_csv(TUNE_OUT / "trials.csv", index=False)

    print(f"\nSelesai. Trials selesai: {len(study.trials)}")
    print(f"Best RMSE (rata-rata {len(tuning_folds)} fold): {study.best_value:.4f}")
    print(f"Best params: {best_hp}")
    print(f"-> {TUNE_OUT / 'best_hyperparams.json'}")


if __name__ == "__main__":
    main()
