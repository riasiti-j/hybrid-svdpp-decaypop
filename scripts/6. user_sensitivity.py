"""Layer 6 entrypoint. For each of the 9 SVD:Pop ratios (SVD_only excluded,
same sweep as the old notebook) x user segment (new/trend/regular, Layer 5)
x protocol (full-catalog, test-only x {val_inner,test_final}), computes
ranking metrics (precision/recall/f1/ndcg), coverage, and RMSE/MAE
(test-only only), then fits a slope/range of each metric vs r_svd across
the ratio sweep per (segment, fold) and tests whether that sensitivity
differs significantly across segments (Friedman omnibus + pairwise
Wilcoxon+Holm post-hoc, 10 folds as paired samples).

Ported/adapted from recsys-project/notebooks/Adaptif Ratio SVDPP
test-only.ipynb Sections 7-10 -- see src/sensitivity.py's docstring for
the adaptation decisions (reuses this pipeline's existing macro-averaged
src/metrics.py formulas rather than the old notebook's micro-averaged
full-catalog precision/recall).

Writes to results/sensitivity/:
    metrics_by_segment_fullcatalog.csv, metrics_by_segment_testonly.csv
    coverage_by_segment_perfold.csv, regression_by_segment_perfold.csv
    sensitivity_slopes_perfold.csv, sensitivity_significance_test.csv

Run from project root (after build_folds.py, generate_hybrid_predictions.py,
build_segments.py):
    python "scripts/6. user_sensitivity.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    N_SPLITS,
    TOP_K,
    RELEVANCE_THRESHOLD,
    RATIOS,
    SENSITIVITY_SEGMENTS,
    SENSITIVITY_MIN_FOLDS,
)
from src.metrics import precision_recall_ndcg, aggregate_per_fold_then_across, rmse_mae, catalogue_coverage
from src.sensitivity import fit_slope_range, segment_sensitivity_test

HYBRID_DIR = RESULTS_DIR / "hybrid"
SEGMENTATION_DIR = RESULTS_DIR / "segmentation"
SENSITIVITY_OUT = RESULTS_DIR / "sensitivity"

RANKING_METRIC_COLS = ["precision", "recall", "f1", "ndcg"]
RATIO_LABELS = list(RATIOS.keys())  # 9 ratios; SVD_only excluded (matches Layer3 sweep + old notebook)


def r_svd_of(ratio_label: str) -> float:
    return RATIOS[ratio_label][0]


def main() -> None:
    user_segments = pd.read_csv(SEGMENTATION_DIR / "user_segments.csv")
    fc_preds_all = pd.read_csv(HYBRID_DIR / "predictions_fullcatalog.csv")
    to_preds_all = pd.read_csv(HYBRID_DIR / "predictions_testonly.csv")

    segment_users = {
        (fold, seg): set(g["user_id"])
        for (fold, seg), g in user_segments.groupby(["fold", "segment"])
    }

    ranking_rows = []
    coverage_rows = []
    regression_rows = []

    for fold_id in range(N_SPLITS):
        fold_dir = PROCESSED_DIR / f"fold{fold_id}"
        train_norm = pd.read_csv(fold_dir / "train_norm.csv")
        val_inner_norm = pd.read_csv(fold_dir / "val_inner_norm.csv")
        test_norm = pd.read_csv(fold_dir / "test_norm.csv")

        catalog = set(train_norm["movie_id"].unique())
        truth_sources_to = {"val_inner": val_inner_norm, "test_final": test_norm}

        fc_preds_fold = fc_preds_all[fc_preds_all["fold"] == fold_id]
        to_preds_fold = to_preds_all[to_preds_all["fold"] == fold_id]

        for segment in SENSITIVITY_SEGMENTS:
            seg_users = segment_users.get((fold_id, segment), set())
            if not seg_users:
                continue

            # --- full-catalog protocol ---
            truth_seg_fc = test_norm[test_norm["user_id"].isin(seg_users)][["user_id", "movie_id", "rating"]]
            for ratio_label in RATIO_LABELS:
                preds_seg = fc_preds_fold[
                    (fc_preds_fold["ratio"] == ratio_label) & (fc_preds_fold["user_id"].isin(seg_users))
                ]

                per_user = precision_recall_ndcg(preds_seg, truth_seg_fc, k=TOP_K, threshold=RELEVANCE_THRESHOLD, mode="fixed_rank")
                if not per_user.empty:
                    per_user = per_user.assign(fold=fold_id, ratio=ratio_label, segment=segment, protocol="fullcatalog", split="n/a")
                    ranking_rows.append(per_user)

                topk_seg = preds_seg[preds_seg["rank"] <= TOP_K]
                coverage_rows.append({
                    "protocol": "fullcatalog", "split": "n/a", "segment": segment,
                    "fold": fold_id, "ratio": ratio_label, "r_svd": r_svd_of(ratio_label),
                    "coverage": catalogue_coverage(topk_seg, catalog),
                })

            # --- test-only protocol (both splits) ---
            for split_name, truth_df in truth_sources_to.items():
                truth_seg_to = truth_df[truth_df["user_id"].isin(seg_users)][["user_id", "movie_id", "rating"]]
                split_preds_fold = to_preds_fold[to_preds_fold["split"] == split_name]

                for ratio_label in RATIO_LABELS:
                    preds_seg = split_preds_fold[
                        (split_preds_fold["ratio"] == ratio_label) & (split_preds_fold["user_id"].isin(seg_users))
                    ]

                    per_user = precision_recall_ndcg(preds_seg, truth_seg_to, k=TOP_K, threshold=RELEVANCE_THRESHOLD, mode="dynamic_score")
                    if not per_user.empty:
                        per_user = per_user.assign(fold=fold_id, ratio=ratio_label, segment=segment, protocol="testonly", split=split_name)
                        ranking_rows.append(per_user)

                    # Coverage denominator = this (fold,ratio,split,segment)'s
                    # own candidate pool size, NOT the full catalog -- the
                    # test-only candidate set is already just this segment's
                    # test items. Ported as-is from the old notebook's
                    # compute_to_coverage; interpret with care, it measures
                    # "fraction of the pool clearing the score threshold,"
                    # not catalog exploration like fc coverage does.
                    candidate_size = preds_seg["movie_id"].nunique()
                    if candidate_size > 0:
                        topk_per_user = preds_seg.sort_values("score", ascending=False).groupby("user_id").head(TOP_K)
                        recommended = topk_per_user[topk_per_user["score"] >= RELEVANCE_THRESHOLD]
                        coverage = recommended["movie_id"].nunique() / candidate_size
                        rmse, mae = rmse_mae(topk_per_user, truth_seg_to)
                        n_matched = topk_per_user.merge(
                            truth_seg_to[["user_id", "movie_id"]], on=["user_id", "movie_id"]
                        ).shape[0]
                    else:
                        coverage = 0.0
                        rmse, mae, n_matched = float("nan"), float("nan"), 0

                    coverage_rows.append({
                        "protocol": "testonly", "split": split_name, "segment": segment,
                        "fold": fold_id, "ratio": ratio_label, "r_svd": r_svd_of(ratio_label),
                        "coverage": coverage,
                    })
                    regression_rows.append({
                        "split": split_name, "segment": segment, "fold": fold_id,
                        "ratio": ratio_label, "r_svd": r_svd_of(ratio_label),
                        "rmse": rmse, "mae": mae, "n_matched": n_matched,
                    })

        print(f"Fold {fold_id}: ratio x segment selesai")

    SENSITIVITY_OUT.mkdir(parents=True, exist_ok=True)

    # --- Step 1: ranking metrics by segment ---
    ranking_all = pd.concat(ranking_rows, ignore_index=True)

    fc_ranking = ranking_all[ranking_all["protocol"] == "fullcatalog"]
    fc_per_fold = fc_ranking.groupby(["segment", "split", "fold", "ratio"])[RANKING_METRIC_COLS].mean().reset_index()
    fc_per_fold["r_svd"] = fc_per_fold["ratio"].map(r_svd_of)
    fc_agg = aggregate_per_fold_then_across(fc_ranking, group_cols=["fold", "ratio", "segment", "split"], metric_cols=RANKING_METRIC_COLS)
    fc_agg["r_svd"] = fc_agg["ratio"].map(r_svd_of)
    fc_agg.to_csv(SENSITIVITY_OUT / "metrics_by_segment_fullcatalog.csv", index=False)

    to_ranking = ranking_all[ranking_all["protocol"] == "testonly"]
    to_per_fold = to_ranking.groupby(["segment", "split", "fold", "ratio"])[RANKING_METRIC_COLS].mean().reset_index()
    to_per_fold["r_svd"] = to_per_fold["ratio"].map(r_svd_of)
    to_agg = aggregate_per_fold_then_across(to_ranking, group_cols=["fold", "ratio", "segment", "split"], metric_cols=RANKING_METRIC_COLS)
    to_agg["r_svd"] = to_agg["ratio"].map(r_svd_of)
    to_agg.to_csv(SENSITIVITY_OUT / "metrics_by_segment_testonly.csv", index=False)

    # --- Step 2/3: coverage & regression per-fold tables ---
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(SENSITIVITY_OUT / "coverage_by_segment_perfold.csv", index=False)

    regression_df = pd.DataFrame(regression_rows)
    regression_df.to_csv(SENSITIVITY_OUT / "regression_by_segment_perfold.csv", index=False)

    # --- Step 4: slope/range fitting, one long table across all metric families ---
    slope_rows = []

    def _fit_group(df: pd.DataFrame, group_cols: list[str], metric: str, protocol: str) -> None:
        for keys, grp in df.groupby(group_cols):
            grp = grp.sort_values("r_svd")
            slope, r2, rng = fit_slope_range(grp["r_svd"].to_numpy(), grp[metric].to_numpy())
            row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
            row.update({"protocol": protocol, "metric": metric, "slope": slope, "r_squared": r2, "range": rng})
            slope_rows.append(row)

    for metric in RANKING_METRIC_COLS:
        _fit_group(fc_per_fold, ["segment", "split", "fold"], metric, "fullcatalog")
        _fit_group(to_per_fold, ["segment", "split", "fold"], metric, "testonly")

    for protocol in ("fullcatalog", "testonly"):
        _fit_group(coverage_df[coverage_df["protocol"] == protocol], ["segment", "split", "fold"], "coverage", protocol)

    for metric in ("rmse", "mae"):
        _fit_group(regression_df, ["segment", "split", "fold"], metric, "testonly")

    slopes_df = pd.DataFrame(slope_rows)[["protocol", "split", "segment", "fold", "metric", "slope", "r_squared", "range"]]
    slopes_df.to_csv(SENSITIVITY_OUT / "sensitivity_slopes_perfold.csv", index=False)

    # --- Step 5: significance test across segments ---
    sig_rows = []
    for (protocol, split, metric), grp in slopes_df.groupby(["protocol", "split", "metric"]):
        for sensitivity_type in ("slope", "range"):
            wide = grp.pivot(index="fold", columns="segment", values=sensitivity_type)
            result = segment_sensitivity_test(wide, SENSITIVITY_SEGMENTS, SENSITIVITY_MIN_FOLDS)
            if result is None:
                print(f"SKIP {protocol}/{split}/{metric}/{sensitivity_type}: fold lengkap < {SENSITIVITY_MIN_FOLDS}, dilewati.")
                continue
            sig_rows.append({"protocol": protocol, "split": split, "metric": metric, "sensitivity_type": sensitivity_type, **result})

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(SENSITIVITY_OUT / "sensitivity_significance_test.csv", index=False)

    # --- Step 6: summary ---
    print("\n=== Rasio terbaik per segmen (test-only, test_final, by ndcg_mean) ===")
    tf = to_agg[to_agg["split"] == "test_final"]
    for segment in SENSITIVITY_SEGMENTS:
        seg_tf = tf[tf["segment"] == segment].sort_values("ndcg_mean", ascending=False)
        if seg_tf.empty:
            continue
        best = seg_tf.iloc[0]
        print(f"  {segment:8s}: rasio terbaik = {best['ratio']} (ndcg_mean={best['ndcg_mean']:.4f})")

    print("\n=== Kombinasi dengan beda sensitivitas antar segmen signifikan (friedman_p < 0.05) ===")
    if not sig_df.empty and (sig_df["friedman_p"] < 0.05).any():
        sig = sig_df[sig_df["friedman_p"] < 0.05]
        print(sig[["protocol", "split", "metric", "sensitivity_type", "n_folds_used", "friedman_p"]].to_string(index=False))
    else:
        print("  (tidak ada)")
    print("\nCatatan: n=10 fold adalah sampel kecil -- baca mean/std sensitivitas (effect size) di")
    print(f"{SENSITIVITY_OUT / 'sensitivity_significance_test.csv'} berdampingan dengan p-value, jangan cuma p-value.")

    print(f"\nSelesai. -> {SENSITIVITY_OUT}")


if __name__ == "__main__":
    main()
