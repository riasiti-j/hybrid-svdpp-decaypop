"""Layer 1 entrypoint. Writes processed/cv10/fold{k}/{train_inner_norm,
val_inner_norm,train_norm,test_norm}.csv, fold_summary.csv, and
manifest.json.

train_norm.csv = train_inner_norm + val_inner_norm concatenated (the full
cleaned train population, for consumers that just need "everything the
model was allowed to see" -- DecayPop pop scores, segmentation features,
full-catalog history exclusion). It carries no separate fit: the z-score
stats baked into its rating_norm column are train_inner's.

test_norm.csv is the fold's ENTIRE test split, never partitioned further.

Run from project root:
    python scripts/build_folds.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    MIN_USER_RATINGS,
    MIN_MOVIE_RATINGS,
    RANDOM_STATE,
    N_SPLITS,
    N_INNER_SPLITS,
    OUTLIER_THRESHOLD,
    PROCESSED_DIR,
)
from src.preprocessing import (
    load_ratings,
    filter_low_frequency,
    create_user_aware_folds,
    remove_outliers_train_only,
    split_train_val_inner,
    zscore_fit_train_inner_apply,
)


def main() -> None:
    ratings = load_ratings()
    ratings_f = filter_low_frequency(ratings)
    print(f"Setelah filter: {len(ratings_f)} baris, {ratings_f['user_id'].nunique()} user, {ratings_f['movie_id'].nunique()} movie")

    fold_assign = create_user_aware_folds(ratings_f, n_splits=N_SPLITS, random_state=RANDOM_STATE)
    ratings_f = ratings_f.reset_index(drop=True)
    ratings_f["fold"] = fold_assign

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for fold_id in range(N_SPLITS):
        train_raw = ratings_f[ratings_f["fold"] != fold_id].drop(columns=["fold"]).reset_index(drop=True)
        test_raw = ratings_f[ratings_f["fold"] == fold_id].drop(columns=["fold"]).reset_index(drop=True)

        train_clean = remove_outliers_train_only(train_raw)

        train_val_split = split_train_val_inner(
            train_clean, n_splits=N_INNER_SPLITS, random_state=RANDOM_STATE + fold_id
        )
        inner_ids = train_val_split[train_val_split["split"] == "train_inner"][["user_id", "movie_id"]]
        val_ids = train_val_split[train_val_split["split"] == "val_inner"][["user_id", "movie_id"]]
        train_inner_raw = train_clean.merge(inner_ids, on=["user_id", "movie_id"])
        val_inner_raw = train_clean.merge(val_ids, on=["user_id", "movie_id"])

        train_inner_norm, val_inner_norm, test_norm = zscore_fit_train_inner_apply(
            train_inner_raw, val_inner_raw, test_raw
        )
        train_norm = pd.concat([train_inner_norm, val_inner_norm], ignore_index=True)

        fold_dir = PROCESSED_DIR / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_inner_norm.to_csv(fold_dir / "train_inner_norm.csv", index=False)
        val_inner_norm.to_csv(fold_dir / "val_inner_norm.csv", index=False)
        train_norm.to_csv(fold_dir / "train_norm.csv", index=False)
        test_norm.to_csv(fold_dir / "test_norm.csv", index=False)

        summary_rows.append(
            {
                "fold": fold_id,
                "train_raw": len(train_raw),
                "train_clean": len(train_clean),
                "outliers_removed": len(train_raw) - len(train_clean),
                "train_inner": len(train_inner_norm),
                "val_inner": len(val_inner_norm),
                "test_raw": len(test_raw),
                "test_norm": len(test_norm),
            }
        )
        print(
            f"Fold {fold_id}: train_inner={len(train_inner_norm):>6} val_inner={len(val_inner_norm):>6} "
            f"({len(train_raw) - len(train_clean)} outlier dibuang dari train), "
            f"test_norm={len(test_norm):>6} -> {fold_dir}"
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PROCESSED_DIR / "fold_summary.csv", index=False)

    manifest = {
        "min_user_ratings": MIN_USER_RATINGS,
        "min_movie_ratings": MIN_MOVIE_RATINGS,
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "outlier_threshold": OUTLIER_THRESHOLD,
        "total_rows_after_filter": len(ratings_f),
        "n_users": int(ratings_f["user_id"].nunique()),
        "n_movies": int(ratings_f["movie_id"].nunique()),
        "max_user_id": int(ratings_f["user_id"].max()),
        "max_movie_id": int(ratings_f["movie_id"].max()),
    }
    with open(PROCESSED_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSelesai. Ringkasan: {PROCESSED_DIR / 'fold_summary.csv'}")
    print(f"Manifest: {PROCESSED_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
