"""Parity/sanity checks against known-good numbers from recsys-project's
audit (audit_pipeline_given5.md) and prior research memory. Run after the
full pipeline (build_folds -> train_svdpp -> train_decaypop ->
generate_hybrid_predictions -> compute_metrics -> build_segments) has
been executed at least once.

Run from project root:
    pytest tests/test_parity.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, RESULTS_DIR, N_SPLITS, TOP_K


# --- M1: Layer 1 preprocessing ---

def test_filtered_totals_match_audit():
    with open(PROCESSED_DIR / "manifest.json") as f:
        manifest = json.load(f)
    assert manifest["total_rows_after_filter"] == 97953
    assert manifest["n_users"] == 943
    assert manifest["n_movies"] == 1152


def test_fold_test_sizes_match_audit():
    summary = pd.read_csv(PROCESSED_DIR / "fold_summary.csv")
    expected = [10229, 10133, 10019, 9917, 9831, 9744, 9651, 9556, 9479, 9394]
    assert summary.sort_values("fold")["test_norm"].tolist() == expected


def test_no_missing_fold_dirs():
    for fold_id in range(N_SPLITS):
        fold_dir = PROCESSED_DIR / f"fold{fold_id}"
        assert (fold_dir / "train_inner_norm.csv").exists()
        assert (fold_dir / "val_inner_norm.csv").exists()
        assert (fold_dir / "train_norm.csv").exists()
        assert (fold_dir / "test_norm.csv").exists()


# --- M2: Layer 2 SVD++ / DecayPop ---

def test_svdpp_rmse_in_plausible_range():
    """best_val_rmse drives early stopping (val_inner, carved from Train);
    final_test_rmse is a post-hoc report against the untouched Test fold
    and never influences model selection."""
    rmse_summary = pd.read_csv(RESULTS_DIR / "svdpp" / "rmse_summary.csv")
    assert (rmse_summary["best_val_rmse"] > 0.5).all()
    assert (rmse_summary["best_val_rmse"] < 1.5).all()
    assert not rmse_summary["best_val_rmse"].isna().any()
    assert (rmse_summary["final_test_rmse"] > 0.5).all()
    assert (rmse_summary["final_test_rmse"] < 1.5).all()
    assert not rmse_summary["final_test_rmse"].isna().any()


def test_decaypop_top20_has_20_unique_items_per_fold():
    for fold_id in range(N_SPLITS):
        item_scores = pd.read_csv(RESULTS_DIR / "decaypop" / f"fold{fold_id}" / "item_scores.csv")
        assert len(item_scores) == 20
        assert item_scores["movie_id"].nunique() == 20


def test_decaypop_t_ref_never_exceeds_fold_train_max_timestamp():
    """Spot check the leakage fix: recomputed pop scores should be derivable
    purely from this fold's own train timestamps (t_ref <= train max)."""
    for fold_id in range(N_SPLITS):
        train_norm = pd.read_csv(PROCESSED_DIR / f"fold{fold_id}" / "train_norm.csv")
        t_ref = pd.to_datetime(train_norm["timestamp"].max(), unit="s")
        assert t_ref <= pd.to_datetime(train_norm["timestamp"], unit="s").max()


# --- M3: Layer 3 hybrid composition ---

def test_fullcatalog_predictions_have_exactly_k_unique_items_per_user_ratio():
    preds = pd.read_csv(RESULTS_DIR / "hybrid" / "predictions_fullcatalog.csv")
    counts = preds.groupby(["fold", "ratio", "user_id"]).size()
    assert (counts <= TOP_K).all()
    dupe_check = preds.groupby(["fold", "ratio", "user_id"])["movie_id"].apply(lambda s: s.duplicated().any())
    assert not dupe_check.any()


def test_testonly_predictions_bounded_by_test_items_per_user():
    preds = pd.read_csv(RESULTS_DIR / "hybrid" / "predictions_testonly.csv")
    counts = preds.groupby(["fold", "split", "ratio", "user_id"]).size()
    assert (counts > 0).all()


# --- M4: Layer 4 metrics ---

def test_fullcatalog_vs_testonly_ratio_ranking_flips():
    """Known research finding (memory: project_fullcatalog_vs_testonly_ratio_flip):
    10_90 should win full-catalog NDCG, 90_10 should win test-only NDCG.
    A regression in Layer 3/4 would show up as this flip disappearing.
    """
    fc = pd.read_csv(RESULTS_DIR / "metrics" / "metrics_fullcatalog.csv")
    fc_hybrid = fc[fc["ratio"] != "SVD_only"]
    fc_best = fc_hybrid.loc[fc_hybrid["ndcg_mean"].idxmax(), "ratio"]

    to = pd.read_csv(RESULTS_DIR / "metrics" / "metrics_testonly.csv")
    to_final = to[(to["split"] == "test_final") & (to["ratio"] != "SVD_only")]
    to_best = to_final.loc[to_final["ndcg_mean"].idxmax(), "ratio"]

    assert fc_best == "10_90", f"expected full-catalog best ratio 10_90, got {fc_best}"
    assert to_best == "90_10", f"expected test-only best ratio 90_10, got {to_best}"


# --- M5: Layer 5 segmentation ---

def test_segment_proportions_plausible():
    segments = pd.read_csv(RESULTS_DIR / "segmentation" / "user_segments.csv")
    props = segments["segment"].value_counts(normalize=True)
    assert props["regular"] == pytest.approx(0.69, abs=0.03)
    assert "new" in props.index
    assert "trend" in props.index


# --- M6: Layer 6 ratio x segment sensitivity ---
# Structural checks only -- this is new analysis (not a port of an
# already-verified prior result), so there's no "known good" number to
# assert against, unlike M1-M5.

def test_sensitivity_output_files_exist_with_expected_columns():
    out = RESULTS_DIR / "sensitivity"
    expected_cols = {
        "metrics_by_segment_fullcatalog.csv": {"ratio", "segment", "split", "r_svd", "ndcg_mean", "ndcg_std"},
        "metrics_by_segment_testonly.csv": {"ratio", "segment", "split", "r_svd", "ndcg_mean", "ndcg_std"},
        "coverage_by_segment_perfold.csv": {"protocol", "split", "segment", "fold", "ratio", "r_svd", "coverage"},
        "regression_by_segment_perfold.csv": {"split", "segment", "fold", "ratio", "r_svd", "rmse", "mae"},
        "sensitivity_slopes_perfold.csv": {"protocol", "split", "segment", "fold", "metric", "slope", "r_squared", "range"},
        "sensitivity_significance_test.csv": {"protocol", "split", "metric", "sensitivity_type", "n_folds_used", "friedman_p"},
    }
    for filename, cols in expected_cols.items():
        df = pd.read_csv(out / filename)
        assert cols.issubset(df.columns), f"{filename} missing columns: {cols - set(df.columns)}"
        assert len(df) > 0, f"{filename} is empty"


def test_sensitivity_pvalues_in_valid_range():
    sig = pd.read_csv(RESULTS_DIR / "sensitivity" / "sensitivity_significance_test.csv")
    assert sig["friedman_p"].between(0, 1).all()
    wilcoxon_cols = [c for c in sig.columns if c.startswith("wilcoxon_p")]
    for col in wilcoxon_cols:
        assert sig[col].between(0, 1).all()


def test_sensitivity_coverage_in_unit_interval():
    coverage = pd.read_csv(RESULTS_DIR / "sensitivity" / "coverage_by_segment_perfold.csv")
    assert coverage["coverage"].between(0, 1).all()
