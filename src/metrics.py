"""Layer 4: evaluation metrics. One parametrized precision_recall_ndcg()
replaces the 6 near-duplicate implementations the audit found across the
old notebooks (compute_accuracy_fold, _surprise_metrics,
compute_fullcatalog_metrics, compute_step6_metrics,
compute_ndcg10_per_user, compute_confusion_per_user) -- the only real
difference between them was one boolean: is top-K cut by fixed rank, or by
a dynamic score threshold.

Relevance is always `rating >= threshold` on the ORIGINAL rating scale
(not rating_norm) -- resolves audit SS13 conflict #1, where DecayPop's
internal evaluation used rating_norm>0 while SVD++/Hybrid used rating>=4;
this pipeline always uses rating>=4.

mode="fixed_rank": denominator = k (for predictions_fullcatalog.csv, which
    always returns exactly k items via slot-filling).
mode="dynamic_score": denominator = count of items in the top-k slice whose
    own `score` column is >= threshold (for predictions_testonly.csv). This
    thresholds the hybrid's blended score directly (both components are
    already on the 1-5 rating scale and r_svd+r_pop=1, so the blend stays
    in [1,5]) rather than the old Mechanism-B-specific "SVD-only est_r"
    check -- an intentional simplification so this single function serves
    both protocols without a hybrid-specific special case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def precision_recall_ndcg(
    preds_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    k: int,
    threshold: float = 4.0,
    mode: str = "fixed_rank",
) -> pd.DataFrame:
    """preds_df: columns user_id, rank, movie_id, score (one fold/ratio/split/protocol group).
    truth_df: columns user_id, movie_id, rating (ground truth for that same group).
    Returns per-user rows; users with zero relevant test items are skipped.
    """
    if mode not in ("fixed_rank", "dynamic_score"):
        raise ValueError(f"unknown mode: {mode}")

    relevant_truth = truth_df[truth_df["rating"] >= threshold]
    relevant_by_user = relevant_truth.groupby("user_id")["movie_id"].apply(set).to_dict()

    rows = []
    for user_id, group in preds_df.groupby("user_id"):
        relevant = relevant_by_user.get(user_id)
        if not relevant:
            continue
        topk = group.sort_values("rank").head(k)
        rec_items = topk["movie_id"].tolist()
        hits = [1 if mid in relevant else 0 for mid in rec_items]
        n_hits = sum(hits)

        if mode == "fixed_rank":
            denom = k
        else:
            denom = int((topk["score"] >= threshold).sum())

        precision = n_hits / denom if denom > 0 else 0.0
        recall = n_hits / len(relevant)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
        idcg = sum(1 / np.log2(idx + 2) for idx in range(min(len(relevant), k)))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        rows.append({
            "user_id": user_id,
            "n_test_relevant": len(relevant),
            "n_recommended": len(rec_items),
            "hits": n_hits,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ndcg": ndcg,
        })
    return pd.DataFrame(rows)


def rmse_mae(preds_df: pd.DataFrame, truth_df: pd.DataFrame) -> tuple[float, float]:
    """RMSE/MAE on the original rating scale, only over (user,movie) pairs
    where both a prediction and a ground-truth rating exist. `score` in
    preds_df must be on the rating scale (SVD-sourced rows only, in
    practice -- callers should filter source=='svd' before calling this
    for predictions_fullcatalog.csv, matching the old convention that
    DecayPop-sourced slots have no rating-scale prediction to compare).
    """
    merged = preds_df.merge(truth_df, on=["user_id", "movie_id"], how="inner")
    if merged.empty:
        return float("nan"), float("nan")
    errors = merged["score"] - merged["rating"]
    return float(np.sqrt((errors ** 2).mean())), float(errors.abs().mean())


def arp_normalized(preds_df: pd.DataFrame, train_item_counts: pd.Series) -> float:
    """ARP_norm = mean_u[ mean_{i in TopK(u)} phi(i) ] / max_i phi(i), phi(i) = train rating count."""
    phi = preds_df["movie_id"].map(train_item_counts).fillna(0)
    per_user = phi.groupby(preds_df["user_id"]).mean()
    max_phi = train_item_counts.max()
    return float(per_user.mean() / max_phi) if max_phi > 0 else 0.0


def catalogue_coverage(preds_df: pd.DataFrame, catalog: set[int]) -> float:
    recommended = set(preds_df["movie_id"].unique())
    return len(recommended) / len(catalog) if catalog else 0.0


def gini_index(preds_df: pd.DataFrame, catalog: set[int]) -> float:
    """Computed over the full catalog, including items recommended zero times."""
    counts = preds_df["movie_id"].value_counts()
    freqs = np.array([counts.get(mid, 0) for mid in catalog], dtype=float)
    freqs.sort()
    n = len(freqs)
    total = freqs.sum()
    if n == 0 or total == 0:
        return 0.0
    j = np.arange(1, n + 1)
    return float((2 * (j * freqs).sum()) / (n * total) - (n + 1) / n)


def aggregate_per_fold_then_across(per_user_metrics: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    """Per-user -> mean per fold (grouped by group_cols incl. 'fold') -> mean+/-std across folds.

    group_cols must include 'fold' plus whatever else identifies a run
    (e.g. ['fold','ratio','split']). Returns one row per non-fold group
    combination with '<metric>_mean' and '<metric>_std' columns.
    """
    per_fold = per_user_metrics.groupby(group_cols)[metric_cols].mean().reset_index()
    across_cols = [c for c in group_cols if c != "fold"]
    agg = per_fold.groupby(across_cols)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    return agg.reset_index()
