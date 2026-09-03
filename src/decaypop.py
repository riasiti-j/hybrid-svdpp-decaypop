"""Layer 2b: DecayPop time-decayed popularity scorer.

Fixes two correctness bugs found while auditing an earlier implementation
of this scoring approach: a t_ref leakage bug (t_ref was once computed
from the whole dataset before the fold loop, so a fold's popularity
scores could end up referenced against a timestamp that actually belongs
to that fold's own test set -- here t_ref is always derived from the
train_fold argument only) and a K=20 bug (generate_recommendations was
once always called with top_k=10 hardcoded, so a later "K=20" evaluation
was silently identical to K=10 -- here top_k is a real parameter).

Formula:
    tm(rating)  = clip((year_ref-year)*12 + (month_ref-month) + 1, 1, DECAYPOP_WINDOW_MONTHS)
    w(tm)       = exp(-tm)
    Pop(i)      = sum over item i's TRAIN ratings of [rating_norm * w(tm)]

DECAYPOP_WINDOW_MONTHS (src/config.py) is this project's own parameter,
not a discard filter -- ratings older than the window still count, just
at the minimum decay weight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DECAYPOP_WINDOW_MONTHS


def _month_offset(ts: pd.Series, t_ref: pd.Timestamp) -> np.ndarray:
    dt = pd.to_datetime(ts, unit="s")
    offset = (t_ref.year - dt.dt.year) * 12 + (t_ref.month - dt.dt.month) + 1
    return offset.clip(lower=1, upper=DECAYPOP_WINDOW_MONTHS).to_numpy()


def compute_pop_scores(train_fold: pd.DataFrame) -> pd.DataFrame:
    """Pop(i) = sum(rating_norm * exp(-tm)) over item i's rows in train_fold.

    t_ref is train_fold['timestamp'].max() -- computed here, from this
    fold's train rows only (the leakage fix).
    """
    t_ref = pd.to_datetime(train_fold["timestamp"].max(), unit="s")
    tm = _month_offset(train_fold["timestamp"], t_ref)
    weight = np.exp(-tm)

    df = train_fold.copy()
    df["weighted"] = df["rating_norm"] * weight
    pop_scores = (
        df.groupby("movie_id")["weighted"]
        .sum()
        .reset_index()
        .rename(columns={"weighted": "pop_score"})
    )
    return pop_scores


def generate_recommendations(pop_scores: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Global Top-K by pop_score (same list for every user -- DecayPop is not personalized).

    Ties broken by movie_id ascending (stable sort on movie_id-sorted input,
    same implicit convention as the old notebook).
    """
    ordered = pop_scores.sort_values(["pop_score", "movie_id"], ascending=[False, True])
    top = ordered.head(top_k).reset_index(drop=True)
    top["rank"] = range(1, len(top) + 1)
    return top


def zscore_normalize_across_folds(pop_scores_per_fold: list[pd.DataFrame]) -> pd.DataFrame:
    """Average pop_score per item across folds, then z-score (ddof=0).

    Consumer: Layer 5 segmentation (Aff_u). Distinct from
    minmax_scale_per_fold below, which serves compose_test_only() in
    Layer 3 -- two different consumers, two different normalizations,
    kept as separately named functions rather than unified.
    """
    all_scores = pd.concat(pop_scores_per_fold, ignore_index=True)
    avg = all_scores.groupby("movie_id")["pop_score"].mean().reset_index()
    mean = avg["pop_score"].mean()
    std = avg["pop_score"].std(ddof=0)
    avg["decaypop_normalized"] = (
        (avg["pop_score"] - mean) / std if std > 0 else 0.0
    )
    return avg.rename(columns={"pop_score": "decaypop_raw"})


def minmax_scale_per_fold(pop_scores: pd.DataFrame) -> pd.DataFrame:
    """Min-max scale this fold's own pop_score to [1,5]. Guards max==min -> 3.0.

    Fold-local (fixes the old bug of reusing a stale Scenario-1 file's
    min/max for every fold). Consumer: compose_test_only() in Layer 3.
    """
    out = pop_scores.copy()
    lo, hi = out["pop_score"].min(), out["pop_score"].max()
    if hi == lo:
        out["pop_norm"] = 3.0
    else:
        out["pop_norm"] = 1 + 4 * (out["pop_score"] - lo) / (hi - lo)
    return out
