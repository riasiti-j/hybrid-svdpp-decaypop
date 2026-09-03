"""Layer 3: hybrid composition. Two protocols, kept as two functions rather
than unified, because they are structurally different evaluations that the
research already confirmed produce a flipped ratio ranking (memory:
project_fullcatalog_vs_testonly_ratio_flip) -- not a bug to reconcile.

compose_full_catalog(): Mechanism A, slot-filling. Item SET changes per
ratio. Scores from SVD++ and DecayPop are never combined into one number --
each source's ranked list is independently sliced into n_svd/n_pop slots.
Ported from hybrid_svdpp_decaypop.ipynb's compose_hybrid() (Step 3).

compose_test_only(): Mechanism B, linear score blend. Item SET is fixed
(the user's test-fold items); only the SCORE changes per ratio via
score = r_svd*svd_est + r_pop*pop_norm. Ported from hybrid_svdpp_decaypop.ipynb
Step 6, with pop_norm now fold-local (see src/decaypop.py:minmax_scale_per_fold)
instead of the old stale-Scenario-1-file normalization.
"""

from __future__ import annotations


def compose_full_catalog(
    user_id: int,
    svd_topn: list[tuple[int, float]],
    pop_topn: list[tuple[int, float]],
    pop_ordered_all: list[tuple[int, float]],
    user_history: set[int],
    n_svd: int,
    n_pop: int,
) -> list[dict]:
    """Slot-fill n_svd items from svd_topn, then n_pop items from
    pop_topn -> pop_ordered_all fallback, excluding history and
    already-chosen items. Returns rows with rank, movie_id, score, source.
    """
    svd_slots = [(mid, score) for mid, score in svd_topn if mid not in user_history][:n_svd]
    chosen = {mid for mid, _ in svd_slots}

    pop_slots: list[tuple[int, float]] = []
    for mid, score in list(pop_topn) + list(pop_ordered_all):
        if len(pop_slots) >= n_pop:
            break
        if mid in chosen or mid in user_history:
            continue
        pop_slots.append((mid, score))
        chosen.add(mid)

    rows = []
    rank = 1
    for mid, score in svd_slots:
        rows.append({"user_id": user_id, "rank": rank, "movie_id": mid, "score": score, "source": "svd"})
        rank += 1
    for mid, score in pop_slots:
        rows.append({"user_id": user_id, "rank": rank, "movie_id": mid, "score": score, "source": "pop"})
        rank += 1
    return rows


def compose_test_only(
    user_id: int,
    test_items: list[int],
    svd_est: dict[int, float],
    pop_norm: dict[int, float],
    r_svd: float,
    r_pop: float,
) -> list[dict]:
    """Re-rank the fixed candidate set `test_items` by a linear score blend.

    svd_est/pop_norm map movie_id -> predicted score for this user's fold
    (both already in the [1,5] scale). Ties broken by descending score,
    stable on input order (matches old Python list.sort behavior).
    """
    scored = []
    for mid in test_items:
        s_svd = svd_est.get(mid, 0.0)
        s_pop = pop_norm.get(mid, 0.0)
        combined = r_svd * s_svd + r_pop * s_pop
        scored.append((mid, combined))
    scored.sort(key=lambda x: x[1], reverse=True)

    rows = []
    for rank, (mid, score) in enumerate(scored, start=1):
        rows.append({"user_id": user_id, "rank": rank, "movie_id": mid, "score": score, "source": "blend"})
    return rows
