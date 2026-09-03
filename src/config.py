"""Single source of frozen configuration for the clean pipeline.

Every value here is either a fixed data-hygiene constant (filtering
thresholds, seed) or a value frozen from prior tuning (Optuna trials for
segmentation thresholds; SVDPP_HYPERPARAMS below is now a fallback
default, not authoritative -- see that constant's comment).
Nothing here should be re-derived or re-pasted in scripts/ or src/ modules
-- import from this module instead. See
C:\\Users\\riacl\\.claude\\plans\\gleaming-riding-badger.md for the audit
trail behind each choice.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "ml-100k"
PROCESSED_DIR = PROJECT_ROOT / "processed" / "cv10"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Layer 1: preprocessing ---
MIN_USER_RATINGS = 5
MIN_MOVIE_RATINGS = 10
RANDOM_STATE = 42
N_SPLITS = 10
OUTLIER_THRESHOLD = 2.0
# Train (9/10 folds) is split again, user-aware round-robin, into
# train_inner/val_inner. 1/N_INNER_SPLITS of train becomes val_inner
# (~10%); the rest is train_inner. Test is NEVER split -- it stays whole
# as the final holdout.
N_INNER_SPLITS = 10

# --- Shared across layers ---
RELEVANCE_THRESHOLD = 4  # rating >= 4 counts as relevant, everywhere
TOP_K = 10

# --- Layer 1a: SVD++ hyperparameter tuning (Optuna) ---
# Search space and fast-CV settings ported from recsys-project/src/svdpp.py::
# tune_hyperparams. Adapted: tuning now reuses the already-built, leak-safe
# train_inner/val_inner split from build_folds.py (the first SVDPP_TUNE_N_FOLDS
# outer folds) instead of deriving a separate ad-hoc split from raw ratings --
# so the tuning criterion matches train_svdpp.py's early-stopping criterion
# and Test is never touched during tuning either.
SVDPP_TUNE_N_TRIALS = 20
SVDPP_TUNE_N_FOLDS = 3
SVDPP_TUNE_N_EPOCHS = 10
SVDPP_TUNE_PATIENCE = 3
SVDPP_TUNE_SEARCH_SPACE = {
    "d": [10, 20, 30, 50],  # categorical
    "alpha": (0.001, 0.05),  # log-uniform
    "lambda": (0.01, 0.20),  # log-uniform
    "decay": (0.80, 0.99),  # uniform
}

# --- Layer 2a: SVD++ ---
# Fallback default, used only if results/svdpp_tuning/best_hyperparams.json
# (written by scripts/1a. tune_svdpp.py) doesn't exist yet. Originally frozen
# from recsys-project Optuna Trial 10 (RMSE=0.9262); scripts/2a. train_svdpp.py
# prefers the tuning script's output when present.
SVDPP_HYPERPARAMS = {
    "d": 30,
    "alpha": 0.023062618121677940,
    "lambda": 0.012502377950801122,
    "decay": 0.9875085179540983,
}
SVDPP_N_EPOCHS = 40
SVDPP_PATIENCE = 5
SVDPP_INIT_SEED = 42
SVDPP_CLIP_MIN = 1.0
SVDPP_CLIP_MAX = 5.0

# --- Layer 2b: DecayPop ---
# Independent parameter choice for this project: cap on tm (month-offset
# since a rating's timestamp), not a discard filter -- ratings older than
# this window still count toward Pop(i), just at the minimum decay weight.
DECAYPOP_WINDOW_MONTHS = 6

# --- Layer 3: hybrid composition ---
# ratio label -> (r_svd, r_pop), written SVD:Pop per recsys-project convention
RATIOS = {
    "10_90": (0.1, 0.9),
    "20_80": (0.2, 0.8),
    "30_70": (0.3, 0.7),
    "40_60": (0.4, 0.6),
    "50_50": (0.5, 0.5),
    "60_40": (0.6, 0.4),
    "70_30": (0.7, 0.3),
    "80_20": (0.8, 0.2),
    "90_10": (0.9, 0.1),
}
SVD_ONLY_RATIO = (1.0, 0.0)

# --- Layer 5: segmentation (frozen, Optuna "PRIMARY" from Adaptif Ratio SVDPP.ipynb Section 8) ---
THETA_NEW_PRIMARY = 23
THETA_TREND_PRIMARY = 82.14

# --- Layer 6: ratio x segment sensitivity ---
SENSITIVITY_SEGMENTS = ("new", "trend", "regular")
SENSITIVITY_MIN_FOLDS = 3  # Friedman/Wilcoxon skipped below this many complete folds
