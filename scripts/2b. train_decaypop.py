"""Layer 2b entrypoint. Computes DecayPop scores per fold, plus the
10-fold-averaged z-score normalization consumed by Layer 5 segmentation.

Run from project root (after scripts/build_folds.py):
    python scripts/train_decaypop.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, RESULTS_DIR, N_SPLITS, TOP_K
from src.decaypop import compute_pop_scores, generate_recommendations, zscore_normalize_across_folds

DECAYPOP_OUT = RESULTS_DIR / "decaypop"
TOP_N = 20  # matches SVD++ top-N so Layer 4 can evaluate K=10 and K=20 for both


def main() -> None:
    pop_scores_per_fold = []

    for fold_id in range(N_SPLITS):
        fold_dir = PROCESSED_DIR / f"fold{fold_id}"
        train_norm = pd.read_csv(fold_dir / "train_norm.csv")

        pop_scores = compute_pop_scores(train_norm)
        recs = generate_recommendations(pop_scores, top_k=TOP_N)

        fold_out = DECAYPOP_OUT / f"fold{fold_id}"
        fold_out.mkdir(parents=True, exist_ok=True)
        pop_scores.to_csv(fold_out / "pop_scores.csv", index=False)
        recs.to_csv(fold_out / "item_scores.csv", index=False)

        pop_scores_per_fold.append(pop_scores)
        print(f"Fold {fold_id}: {len(pop_scores)} item punya pop_score, top-{TOP_N} tersimpan -> {fold_out}")

    normalized = zscore_normalize_across_folds(pop_scores_per_fold)
    normalized.to_csv(DECAYPOP_OUT / "decaypop_normalized.csv", index=False)
    print(f"\nSelesai. decaypop_normalized.csv: {len(normalized)} item -> {DECAYPOP_OUT / 'decaypop_normalized.csv'}")


if __name__ == "__main__":
    main()
