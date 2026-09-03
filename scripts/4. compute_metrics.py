"""Layer 4 entrypoint. Reads predictions_fullcatalog.csv / predictions_testonly.csv,
applies src.metrics.precision_recall_ndcg with the mode matching each
protocol, aggregates per-user -> per-fold -> mean+/-std across folds, and
writes:

    results/metrics/metrics_fullcatalog.csv   (mode="fixed_rank", truth=whole test fold)
    results/metrics/metrics_testonly.csv      (mode="dynamic_score", truth=val_inner/test_final split)

Run from project root (after generate_hybrid_predictions.py):
    python scripts/compute_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, RESULTS_DIR, N_SPLITS, TOP_K, RELEVANCE_THRESHOLD
from src.metrics import precision_recall_ndcg, aggregate_per_fold_then_across

HYBRID_DIR = RESULTS_DIR / "hybrid"
METRICS_OUT = RESULTS_DIR / "metrics"

METRIC_COLS = ["precision", "recall", "f1", "ndcg"]


def main() -> None:
    METRICS_OUT.mkdir(parents=True, exist_ok=True)

    # --- Full-catalog: mode=fixed_rank, truth = the whole fold's test_norm ---
    fc_preds = pd.read_csv(HYBRID_DIR / "predictions_fullcatalog.csv")
    fc_rows = []
    for fold_id in range(N_SPLITS):
        truth = pd.read_csv(PROCESSED_DIR / f"fold{fold_id}" / "test_norm.csv")[["user_id", "movie_id", "rating"]]
        fold_preds = fc_preds[fc_preds["fold"] == fold_id]
        for ratio_label, group in fold_preds.groupby("ratio"):
            per_user = precision_recall_ndcg(group, truth, k=TOP_K, threshold=RELEVANCE_THRESHOLD, mode="fixed_rank")
            per_user["fold"] = fold_id
            per_user["ratio"] = ratio_label
            fc_rows.append(per_user)
    fc_per_user = pd.concat(fc_rows, ignore_index=True)
    fc_agg = aggregate_per_fold_then_across(fc_per_user, group_cols=["fold", "ratio"], metric_cols=METRIC_COLS)
    fc_agg.to_csv(METRICS_OUT / "metrics_fullcatalog.csv", index=False)
    print("=== Full-catalog: ranking rasio by ndcg_mean ===")
    print(fc_agg.sort_values("ndcg_mean", ascending=False)[["ratio", "precision_mean", "ndcg_mean"]].to_string(index=False))

    # --- Test-only: mode=dynamic_score, truth = val_inner/test_final split ---
    to_preds = pd.read_csv(HYBRID_DIR / "predictions_testonly.csv")
    to_rows = []
    for fold_id in range(N_SPLITS):
        # val_inner truth comes from Train's held-out val_inner_norm.csv;
        # test_final truth is the whole, never-split test_norm.csv.
        val_inner_norm = pd.read_csv(PROCESSED_DIR / f"fold{fold_id}" / "val_inner_norm.csv")
        test_norm = pd.read_csv(PROCESSED_DIR / f"fold{fold_id}" / "test_norm.csv")
        truth_sources = {"val_inner": val_inner_norm, "test_final": test_norm}

        fold_preds = to_preds[to_preds["fold"] == fold_id]
        for split_name, truth_df in truth_sources.items():
            truth = truth_df[["user_id", "movie_id", "rating"]]
            split_preds = fold_preds[fold_preds["split"] == split_name]
            for ratio_label, group in split_preds.groupby("ratio"):
                per_user = precision_recall_ndcg(group, truth, k=TOP_K, threshold=RELEVANCE_THRESHOLD, mode="dynamic_score")
                per_user["fold"] = fold_id
                per_user["ratio"] = ratio_label
                per_user["split"] = split_name
                to_rows.append(per_user)
    to_per_user = pd.concat(to_rows, ignore_index=True)
    to_agg = aggregate_per_fold_then_across(to_per_user, group_cols=["fold", "ratio", "split"], metric_cols=METRIC_COLS)
    to_agg.to_csv(METRICS_OUT / "metrics_testonly.csv", index=False)
    print("\n=== Test-only (test_final): ranking rasio by ndcg_mean ===")
    tf = to_agg[to_agg["split"] == "test_final"]
    print(tf.sort_values("ndcg_mean", ascending=False)[["ratio", "precision_mean", "ndcg_mean"]].to_string(index=False))

    print(f"\nSelesai. -> {METRICS_OUT}")


if __name__ == "__main__":
    main()
