"""Layer 5 entrypoint. Writes results/segmentation/user_segments.csv
(fold, user_id, segment, n_u, aff_pct_u).

Run from project root (after build_folds.py and train_decaypop.py):
    python scripts/build_segments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PROCESSED_DIR, RESULTS_DIR, N_SPLITS, THETA_NEW_PRIMARY, THETA_TREND_PRIMARY
from src.segmentation import compute_n_u, compute_affinity, compute_affinity_percentile, classify_users

SEGMENTATION_OUT = RESULTS_DIR / "segmentation"


def main() -> None:
    decaypop_normalized = pd.read_csv(RESULTS_DIR / "decaypop" / "decaypop_normalized.csv")
    mean_raw = decaypop_normalized["decaypop_raw"].mean()
    std_raw = decaypop_normalized["decaypop_raw"].std(ddof=0)
    cold_start_score = float((0 - mean_raw) / std_raw) if std_raw > 0 else 0.0

    all_rows = []
    for fold_id in range(N_SPLITS):
        train_norm = pd.read_csv(PROCESSED_DIR / f"fold{fold_id}" / "train_norm.csv")

        n_u = compute_n_u(train_norm)
        aff = compute_affinity(train_norm, decaypop_normalized, cold_start_score)
        aff_pct = compute_affinity_percentile(aff)
        segments = classify_users(n_u, aff_pct, THETA_NEW_PRIMARY, THETA_TREND_PRIMARY)
        segments.insert(0, "fold", fold_id)
        all_rows.append(segments)

        counts = segments["segment"].value_counts().to_dict()
        print(f"Fold {fold_id}: {counts}")

    result = pd.concat(all_rows, ignore_index=True)
    SEGMENTATION_OUT.mkdir(parents=True, exist_ok=True)
    result.to_csv(SEGMENTATION_OUT / "user_segments.csv", index=False)
    print(f"\nSelesai. {len(result)} baris -> {SEGMENTATION_OUT / 'user_segments.csv'}")
    print(result["segment"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
