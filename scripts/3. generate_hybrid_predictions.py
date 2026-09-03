"""Layer 3 entrypoint. For each fold x 9 ratios (+ SVD_only baseline) x
{full_catalog, test_only} x {val_inner, test_final}, composes hybrid
recommendation lists and writes two CSVs:

    results/hybrid/predictions_fullcatalog.csv
    results/hybrid/predictions_testonly.csv

Both share the schema: fold, user_id, ratio, split, rank, movie_id, score, source.

Mechanism B's "val_inner" items come from Train's held-out val_inner_norm.csv;
"test_final" items are the fold's entire, never-split test_norm.csv.

Run from project root (after build_folds.py, train_svdpp.py, train_decaypop.py):
    python scripts/generate_hybrid_predictions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, RESULTS_DIR, N_SPLITS, TOP_K, RATIOS, SVD_ONLY_RATIO
from src.svdpp import load_model, predict_batch
from src.decaypop import minmax_scale_per_fold
from src.hybrid import compose_full_catalog, compose_test_only

SVDPP_DIR = RESULTS_DIR / "svdpp"
DECAYPOP_DIR = RESULTS_DIR / "decaypop"
HYBRID_OUT = RESULTS_DIR / "hybrid"

ALL_RATIOS = {**RATIOS, "SVD_only": SVD_ONLY_RATIO}


def n_svd_n_pop(r_svd: float) -> tuple[int, int]:
    n_svd = round(TOP_K * r_svd)
    return n_svd, TOP_K - n_svd


def main() -> None:
    HYBRID_OUT.mkdir(parents=True, exist_ok=True)
    fullcatalog_rows = []
    testonly_rows = []

    for fold_id in range(N_SPLITS):
        fold_dir = PROCESSED_DIR / f"fold{fold_id}"
        train_norm = pd.read_csv(fold_dir / "train_norm.csv")
        # val_inner comes from Train (never Test); test_final is the whole,
        # never-split Test fold.
        val_inner_norm = pd.read_csv(fold_dir / "val_inner_norm.csv")
        test_norm_full = pd.read_csv(fold_dir / "test_norm.csv")
        split_sources = {"val_inner": val_inner_norm, "test_final": test_norm_full}

        # --- Layer 2 artifacts for this fold ---
        top20 = pd.read_csv(SVDPP_DIR / f"fold{fold_id}" / "top20_unseen.csv")
        svd_topn_by_user = {
            uid: list(zip(g.sort_values("rank")["movie_id"], g.sort_values("rank")["svdpp_score"]))
            for uid, g in top20.groupby("user_id")
        }
        params, impl_N, impl_inv = load_model(SVDPP_DIR / f"fold{fold_id}" / "model.pkl")

        pop_scores = pd.read_csv(DECAYPOP_DIR / f"fold{fold_id}" / "pop_scores.csv")
        pop_recs = pd.read_csv(DECAYPOP_DIR / f"fold{fold_id}" / "item_scores.csv")
        pop_topn_all_users = list(zip(pop_recs.sort_values("rank")["movie_id"], pop_recs.sort_values("rank")["pop_score"]))
        pop_ordered_all = list(zip(
            pop_scores.sort_values(["pop_score", "movie_id"], ascending=[False, True])["movie_id"],
            pop_scores.sort_values(["pop_score", "movie_id"], ascending=[False, True])["pop_score"],
        ))
        pop_norm_df = minmax_scale_per_fold(pop_scores)
        pop_norm_map = pop_norm_df.set_index("movie_id")["pop_norm"].to_dict()

        history_by_user = train_norm.groupby("user_id")["movie_id"].apply(set).to_dict()

        # --- Mechanism A: full-catalog slot-filling ---
        for ratio_label, (r_svd, r_pop) in ALL_RATIOS.items():
            n_svd, n_pop = n_svd_n_pop(r_svd)
            for user_id, history in history_by_user.items():
                svd_topn = svd_topn_by_user.get(user_id, [])
                rows = compose_full_catalog(
                    user_id, svd_topn, pop_topn_all_users, pop_ordered_all, history, n_svd, n_pop
                )
                for row in rows:
                    row["fold"] = fold_id
                    row["ratio"] = ratio_label
                    row["split"] = "n/a"  # full-catalog is not split-scoped; same rec list feeds both
                    fullcatalog_rows.append(row)

        # --- Mechanism B: test-only re-ranking, per split ---
        for split_name, split_df in split_sources.items():
            items_by_user = split_df.groupby("user_id")["movie_id"].apply(list).to_dict()

            for user_id, test_items in items_by_user.items():
                svd_est = {mid: score for mid, score in predict_batch(user_id, test_items, params, impl_N, impl_inv)}
                for ratio_label, (r_svd, r_pop) in ALL_RATIOS.items():
                    rows = compose_test_only(user_id, test_items, svd_est, pop_norm_map, r_svd, r_pop)
                    for row in rows:
                        row["fold"] = fold_id
                        row["ratio"] = ratio_label
                        row["split"] = split_name
                        testonly_rows.append(row)

        print(f"Fold {fold_id}: full-catalog {len(history_by_user)} user x {len(ALL_RATIOS)} rasio selesai")

    cols = ["fold", "ratio", "split", "user_id", "rank", "movie_id", "score", "source"]
    pd.DataFrame(fullcatalog_rows)[cols].to_csv(HYBRID_OUT / "predictions_fullcatalog.csv", index=False)
    pd.DataFrame(testonly_rows)[cols].to_csv(HYBRID_OUT / "predictions_testonly.csv", index=False)
    print(f"\nSelesai. {len(fullcatalog_rows)} baris -> predictions_fullcatalog.csv")
    print(f"Selesai. {len(testonly_rows)} baris -> predictions_testonly.csv")


if __name__ == "__main__":
    main()
