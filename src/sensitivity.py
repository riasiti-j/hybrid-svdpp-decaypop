"""Layer 6: ratio x segment sensitivity analysis.

Answers a question Layer 4 (per-ratio metrics) and Layer 5 (per-user
segments) never connect: which SVD:Pop ratio is best for which user
segment (new/trend/regular), and is the ratio's effect on a metric
significantly different across segments?

Ported/adapted from recsys-project/notebooks/Adaptif Ratio SVDPP
test-only.ipynb, Section 8 (slope/range sensitivity + Friedman/Wilcoxon
significance test across segments). The metric formulas themselves
(precision/recall/f1/ndcg, rmse/mae) are NOT re-ported here -- Layer 6
reuses src/metrics.py::precision_recall_ndcg/rmse_mae/catalogue_coverage
so this analysis stays on the one metric convention the rest of this
pipeline already uses, instead of reintroducing the old notebook's
micro-averaged full-catalog precision/recall (one of the near-duplicate
formulas src/metrics.py was built to retire).

Caveat carried over verbatim from the old notebook: n=10 folds is a small
paired sample for Friedman/Wilcoxon -- report effect size (mean/std of
slope or range) alongside the p-value, not the p-value alone.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, linregress, wilcoxon


def fit_slope_range(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """slope & r_squared (scipy.stats.linregress) + range (max-min) of y vs x.

    NaN for all three if fewer than 2 points or any NaN in y. (0.0, 0.0, 0.0)
    if y is constant (linregress is undefined for zero variance) -- both
    edge cases replicate the old notebook's fitting loop exactly.
    """
    if len(x) < 2 or np.any(np.isnan(y)):
        return float("nan"), float("nan"), float("nan")
    if np.all(y == y[0]):
        return 0.0, 0.0, 0.0
    res = linregress(x, y)
    return float(res.slope), float(res.rvalue ** 2), float(np.max(y) - np.min(y))


def holm_correction(pvals: list[float]) -> np.ndarray:
    """Step-down Holm-Bonferroni correction.

    Ported verbatim from Adaptif Ratio SVDPP test-only.ipynb's
    holm_correct() (a from-scratch implementation, not statsmodels).
    """
    pvals = np.array(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    corrected = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        corrected[idx] = min(running_max, 1.0)
    return corrected


def segment_sensitivity_test(
    wide: pd.DataFrame,
    segments: tuple[str, ...],
    min_folds: int,
) -> dict | None:
    """Friedman (omnibus) + pairwise Wilcoxon+Holm (post-hoc) across segments.

    wide: index=fold, columns=segments, values=slope or range for one
    (protocol, split, metric, sensitivity_type) combination.

    Returns None if fewer than min_folds complete (non-NaN-across-all-
    segments) rows remain -- matches the old notebook's "terlalu sedikit
    fold untuk uji" skip.
    """
    wide = wide.dropna()
    n_folds_used = len(wide)
    if n_folds_used < min_folds:
        return None

    samples = [wide[seg].to_numpy() for seg in segments]
    chi2, p_omni = friedmanchisquare(*samples)

    pairs = list(combinations(segments, 2))
    raw_pvals = []
    for a, b in pairs:
        diff = wide[a] - wide[b]
        if np.all(diff == 0):
            raw_pvals.append(1.0)
        else:
            raw_pvals.append(float(wilcoxon(wide[a], wide[b]).pvalue))
    holm_pvals = holm_correction(raw_pvals)

    row: dict = {
        "n_folds_used": n_folds_used,
        "friedman_chi2": float(chi2),
        "friedman_p": float(p_omni),
    }
    for seg in segments:
        row[f"mean_{seg}"] = float(wide[seg].mean())
        row[f"std_{seg}"] = float(wide[seg].std())
    for (a, b), p_raw, p_holm in zip(pairs, raw_pvals, holm_pvals):
        row[f"wilcoxon_p_{a}_vs_{b}"] = p_raw
        row[f"wilcoxon_p_holm_{a}_vs_{b}"] = float(p_holm)
    return row
