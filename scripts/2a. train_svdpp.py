"""Layer 2a entrypoint. Trains SVD++ per fold, saves model.pkl + top20_unseen.csv.

Trains on train_inner only; early stopping selects the best epoch on
val_inner (both carved from Train). Test is touched exactly once, after
training is finished, purely to report final_test_rmse -- it never
influences model selection.

Hyperparameters (d, alpha, lambda, decay) are read from
results/svdpp_tuning/best_hyperparams.json, written by
scripts/1a. tune_svdpp.py. If that file doesn't exist yet, falls back to
the frozen SVDPP_HYPERPARAMS default in src/config.py.

Run from project root (after scripts/build_folds.py, ideally after
scripts/1a. tune_svdpp.py):
    python scripts/train_svdpp.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    N_SPLITS,
    SVDPP_HYPERPARAMS,
    SVDPP_N_EPOCHS,
    SVDPP_PATIENCE,
    SVDPP_INIT_SEED,
)
from src.svdpp import (
    init_params,
    build_implicit_feedback,
    train_svdpp,
    compute_rmse_mae,
    recommend_topn,
    save_model,
)

SVDPP_OUT = RESULTS_DIR / "svdpp"
TUNED_HYPERPARAMS_PATH = RESULTS_DIR / "svdpp_tuning" / "best_hyperparams.json"
TOP_N = 20


def load_hyperparams() -> dict:
    if TUNED_HYPERPARAMS_PATH.exists():
        with open(TUNED_HYPERPARAMS_PATH) as f:
            tuned = json.load(f)
        hp = {"d": tuned["d"], "alpha": tuned["alpha"], "lambda": tuned["lambda"], "decay": tuned["decay"]}
        print(f"Pakai hyperparameter hasil tuning ({TUNED_HYPERPARAMS_PATH}): {hp}")
        return hp
    print(
        f"[!] {TUNED_HYPERPARAMS_PATH} tidak ditemukan -- pakai SVDPP_HYPERPARAMS beku "
        f"dari src/config.py. Jalankan 'scripts/1a. tune_svdpp.py' dulu untuk tuning."
    )
    return dict(SVDPP_HYPERPARAMS)


def main() -> None:
    with open(PROCESSED_DIR / "manifest.json") as f:
        manifest = json.load(f)
    n_users_max_id = manifest["max_user_id"]
    n_items_max_id = manifest["max_movie_id"]

    hp_base = load_hyperparams()
    rmse_rows = []

    for fold_id in range(N_SPLITS):
        fold_out_check = SVDPP_OUT / f"fold{fold_id}"
        if (fold_out_check / "model.pkl").exists() and (fold_out_check / "top20_unseen.csv").exists():
            print(f"Fold {fold_id}: sudah selesai (model.pkl + top20_unseen.csv ada) -> skip")
            history = pd.read_csv(fold_out_check / "training_history.csv")
            rmse_rows.append({
                "fold": fold_id,
                "best_val_rmse": history["val_rmse"].min(),
                "final_test_rmse": None,
                "epochs_run": len(history),
            })
            continue

        fold_dir = PROCESSED_DIR / f"fold{fold_id}"
        train_norm = pd.read_csv(fold_dir / "train_norm.csv")
        train_inner_norm = pd.read_csv(fold_dir / "train_inner_norm.csv")
        val_inner_norm = pd.read_csv(fold_dir / "val_inner_norm.csv")
        test_norm = pd.read_csv(fold_dir / "test_norm.csv")

        train_ratings = list(zip(train_inner_norm["user_id"], train_inner_norm["movie_id"], train_inner_norm["rating"]))
        val_ratings = list(zip(val_inner_norm["user_id"], val_inner_norm["movie_id"], val_inner_norm["rating"]))
        test_ratings = list(zip(test_norm["user_id"], test_norm["movie_id"], test_norm["rating"]))
        mu = float(train_inner_norm["rating"].mean())

        params = init_params(mu, n_users_max_id, n_items_max_id, d=hp_base["d"], seed=SVDPP_INIT_SEED)
        impl_N, impl_inv = build_implicit_feedback(train_ratings)
        hp = copy.deepcopy(hp_base)

        print(f"--- Fold {fold_id} ---")
        best_params, history = train_svdpp(
            train_ratings, val_ratings, params, hp, impl_N, impl_inv,
            n_epochs=SVDPP_N_EPOCHS, patience=SVDPP_PATIENCE,
            shuffle_seed=SVDPP_INIT_SEED + fold_id, verbose=True,
        )
        best_val_rmse = min(history["val_rmse"])
        final_test_rmse, _ = compute_rmse_mae(test_ratings, best_params, impl_N, impl_inv)
        rmse_rows.append({
            "fold": fold_id,
            "best_val_rmse": best_val_rmse,
            "final_test_rmse": final_test_rmse,
            "epochs_run": len(history["epoch"]),
        })

        fold_out = SVDPP_OUT / f"fold{fold_id}"
        save_model(fold_out / "model.pkl", best_params, impl_N, impl_inv)
        pd.DataFrame(history).to_csv(fold_out / "training_history.csv", index=False)

        # Recs must exclude everything the user rated anywhere in Train
        # (train_inner + val_inner), even though the model itself was only
        # fit on train_inner -- val_inner ratings are still real, known
        # history from the user's point of view.
        history_users = train_norm.groupby("user_id")["movie_id"].apply(set).to_dict()
        rows = []
        for user_id in sorted(train_norm["user_id"].unique()):
            history_set = history_users.get(user_id, set())
            topn = recommend_topn(best_params, impl_N, impl_inv, user_id, n_items_max_id, history_set, TOP_N)
            for rank, (movie_id, score) in enumerate(topn, start=1):
                rows.append({"user_id": user_id, "movie_id": movie_id, "svdpp_score": score, "rank": rank})
        pd.DataFrame(rows).to_csv(fold_out / "top20_unseen.csv", index=False)
        print(f"Fold {fold_id}: best_val_rmse={best_val_rmse:.4f} final_test_rmse={final_test_rmse:.4f} -> {fold_out}")

    pd.DataFrame(rmse_rows).to_csv(SVDPP_OUT / "rmse_summary.csv", index=False)
    print(f"\nSelesai. Ringkasan RMSE: {SVDPP_OUT / 'rmse_summary.csv'}")


if __name__ == "__main__":
    main()
