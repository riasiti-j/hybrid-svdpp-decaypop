"""Layer 5: user segmentation (new / trend / regular).

Ported from Adaptif Ratio SVDPP.ipynb Sections 2-5. Thresholds are frozen
constants imported from src.config (THETA_NEW_PRIMARY, THETA_TREND_PRIMARY)
rather than re-pasted literals -- resolves audit finding #5 (thresholds
existed only as hardcoded numbers inside notebook cells, duplicated across
two notebooks).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_n_u(train_ratings: pd.DataFrame) -> dict[int, int]:
    """n_u = number of ratings user u has in train."""
    return train_ratings.groupby("user_id").size().to_dict()


def compute_affinity(
    train_ratings: pd.DataFrame,
    decaypop_normalized: pd.DataFrame,
    cold_start_score: float,
) -> dict[int, float]:
    """Aff_u = mean decaypop_normalized score of items user u rated in train.

    cold_start_score is used as a fallback when a rated item has no
    decaypop_normalized entry (should not happen given both are derived
    from the same filtered catalog, but guarded defensively).
    """
    score_map = decaypop_normalized.set_index("movie_id")["decaypop_normalized"].to_dict()
    aff = {}
    for user_id, group in train_ratings.groupby("user_id"):
        scores = [score_map.get(mid, cold_start_score) for mid in group["movie_id"]]
        aff[user_id] = float(np.mean(scores)) if scores else cold_start_score
    return aff


def compute_affinity_percentile(affinity: dict[int, float]) -> dict[int, float]:
    """Aff_pct_u = percentile rank of Aff_u among all users, 0-100."""
    series = pd.Series(affinity)
    pct = series.rank(pct=True, method="average") * 100
    return pct.to_dict()


def classify_users(
    n_u: dict[int, int],
    aff_pct: dict[int, float],
    theta_new: float,
    theta_trend: float,
) -> pd.DataFrame:
    """Precedence: new (n_u < theta_new) checked first, then trend
    (aff_pct >= theta_trend among the rest), else regular.
    """
    user_ids = sorted(n_u.keys())
    n_arr = np.array([n_u[u] for u in user_ids])
    aff_arr = np.array([aff_pct.get(u, 0.0) for u in user_ids])

    is_new = n_arr < theta_new
    is_trend = (~is_new) & (aff_arr >= theta_trend)
    segment = np.where(is_new, "new", np.where(is_trend, "trend", "regular"))

    return pd.DataFrame({
        "user_id": user_ids,
        "n_u": n_arr,
        "aff_pct_u": aff_arr,
        "segment": segment,
    })
