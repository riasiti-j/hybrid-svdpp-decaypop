"""Layer 1: filter -> user-aware 10-fold split -> train-only outlier removal
-> train_inner/val_inner split -> z-score fit on train_inner only, applied
to train_inner/val_inner/test.

Test is a strict, never-split final holdout: it is filtered to users seen
in train_inner and transformed with train_inner's frozen per-user stats,
but never used to fit anything and never partitioned. Validation
(val_inner) is carved out of Train instead, so model selection (SVD++
early stopping, ratio selection in Layer 4) never touches Test.

Test-user-not-in-train_inner handling: rows are dropped (not given a
global-mean fallback). This matches export_cv10_folds.py's original
behavior (recsys-project/scripts/export_cv10_folds.py, confirmed by the
pipeline audit audit_pipeline_given5.md as the one part of the old
codebase that already behaved like a single-source-of-truth) -- not the
alternate fallback-to-mean convention used in svd++ comparison.ipynb. Kept
for fidelity to the verified precedent rather than introducing an
untested convention switch.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import (
    DATA_DIR,
    MIN_USER_RATINGS,
    MIN_MOVIE_RATINGS,
    OUTLIER_THRESHOLD,
)


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(
        DATA_DIR / "u.data",
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )


def filter_low_frequency(ratings: pd.DataFrame) -> pd.DataFrame:
    user_counts = ratings["user_id"].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    ratings_f = ratings[ratings["user_id"].isin(valid_users)].copy()

    movie_counts = ratings_f["movie_id"].value_counts()
    valid_movies = movie_counts[movie_counts >= MIN_MOVIE_RATINGS].index
    ratings_f = ratings_f[ratings_f["movie_id"].isin(valid_movies)].copy()
    return ratings_f.reset_index(drop=True)


def create_user_aware_folds(df: pd.DataFrame, n_splits: int, random_state: int) -> np.ndarray:
    """Round-robin fold assignment within each user's ratings, independently shuffled."""
    rng = np.random.RandomState(random_state)
    df = df.reset_index(drop=True)
    fold_assign = np.full(len(df), -1, dtype=int)
    for _, grp in df.groupby("user_id"):
        idxs = grp.index.tolist()
        shuffled = rng.permutation(idxs)
        for i, idx in enumerate(shuffled):
            fold_assign[idx] = i % n_splits
    return fold_assign


def remove_outliers_train_only(train_raw: pd.DataFrame) -> pd.DataFrame:
    stats = (
        train_raw.groupby("user_id")["rating"]
        .agg(user_mean="mean", user_std="std")
        .reset_index()
    )
    stats["user_std"] = stats["user_std"].fillna(0)

    chk = train_raw.merge(stats, on="user_id")
    chk["is_outlier"] = (
        (chk["rating"] - chk["user_mean"]).abs() > OUTLIER_THRESHOLD * chk["user_std"]
    )
    return (
        chk[~chk["is_outlier"]]
        .drop(columns=["user_mean", "user_std", "is_outlier"])
        .reset_index(drop=True)
    )


def split_train_val_inner(
    train_clean: pd.DataFrame, n_splits: int, random_state: int
) -> pd.DataFrame:
    """Split train_clean (never test!) into train_inner/val_inner.

    Uses the same user-aware round-robin logic as create_user_aware_folds
    so every user contributes to both buckets in proportion (~1/n_splits
    goes to val_inner). Returns fold-scoped rows: user_id, movie_id, split.
    """
    assignment = create_user_aware_folds(train_clean, n_splits=n_splits, random_state=random_state)
    out = train_clean[["user_id", "movie_id"]].copy()
    out["split"] = np.where(assignment == 0, "val_inner", "train_inner")
    return out.reset_index(drop=True)


def zscore_fit_train_inner_apply(
    train_inner: pd.DataFrame, val_inner: pd.DataFrame, test_raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit per-user mean/std (ddof=1, pandas default) on train_inner ONLY,
    then transform train_inner, val_inner, and test with those frozen
    stats. Val and Test never contribute to the fit -- only train_inner
    does -- so neither contaminates the statistics used to normalize
    itself.
    """
    stats = (
        train_inner.groupby("user_id")["rating"]
        .agg(user_mean="mean", user_std="std")
        .reset_index()
    )
    stats["user_std"] = stats["user_std"].fillna(0)
    users_in_train_inner = stats["user_id"].unique()

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        filtered = df[df["user_id"].isin(users_in_train_inner)].copy()
        merged = filtered.merge(stats, on="user_id", how="left")
        merged["rating_norm"] = np.where(
            merged["user_std"] > 0,
            (merged["rating"] - merged["user_mean"]) / merged["user_std"],
            0.0,
        )
        return merged[
            ["user_id", "movie_id", "rating", "rating_norm", "timestamp"]
        ].reset_index(drop=True)

    return _transform(train_inner), _transform(val_inner), _transform(test_raw)
