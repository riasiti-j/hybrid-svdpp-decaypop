"""Layer 2a: SVD++ (implicit feedback) trainer.

Ported from recsys-project/src/svdpp.py's core numeric routines (confirmed
by the pipeline audit as the actively-used, formula-verified implementation)
-- not the rest of that module, which belongs to a superseded 3-arm
architecture (personalized/popularity/exploration + Thompson sampling) that
the audit found to be dead code relative to the actual SVD++/DecayPop/
Hybrid research line.

Formula: r_hat(u,i) = clip(mu + b_u + b_i + q_i . (p_u + |N(u)|^-0.5 * sum_j y_j), 1, 5)
where N(u) = items rated by u in train (implicit feedback).

Deviation from the old module: SGD shuffle order is now explicitly seeded
(np.random.default_rng(seed=42+fold)) instead of relying on the unseeded
global `random` module, per the audit's reproducibility recommendation
(audit_pipeline_given5.md SS5). This is the only behavioral change; the
update formulas and clipping range are unchanged.
"""

from __future__ import annotations

import copy
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SVDPP_CLIP_MIN, SVDPP_CLIP_MAX


def init_params(mu: float, n_users: int, n_items: int, d: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "mu": mu,
        "bu": np.zeros(n_users + 1),
        "bi": np.zeros(n_items + 1),
        "pu": rng.normal(0, 0.01, (n_users + 1, d)),
        "qi": rng.normal(0, 0.01, (n_items + 1, d)),
        "yj": rng.normal(0, 0.01, (n_items + 1, d)),
    }


def build_implicit_feedback(train_ratings: list[tuple[int, int, float]]):
    impl_N = defaultdict(list)
    for u, i, _ in train_ratings:
        impl_N[u].append(i)
    impl_N = dict(impl_N)
    impl_inv = {u: 1.0 / math.sqrt(len(v)) for u, v in impl_N.items()}
    return impl_N, impl_inv


def predict_batch(user_id, item_ids, params, impl_N, impl_inv):
    mu = params["mu"]
    bu = params["bu"][user_id]
    pu = params["pu"][user_id]
    N_u = impl_N.get(user_id, [])
    implicit_factor = (
        impl_inv[user_id] * params["yj"][N_u].sum(axis=0)
        if N_u
        else np.zeros(params["yj"].shape[1])
    )
    user_vec = pu + implicit_factor
    item_ids = list(item_ids)
    r_hats = np.clip(
        mu + bu + params["bi"][item_ids] + params["qi"][item_ids].dot(user_vec),
        SVDPP_CLIP_MIN,
        SVDPP_CLIP_MAX,
    )
    return list(zip(item_ids, r_hats.tolist()))


def train_one_epoch(train_ratings, params, hyperparams, impl_N, impl_inv, rng: np.random.Generator):
    alpha = hyperparams["alpha"]
    lam = hyperparams["lambda"]
    bu, bi, pu, qi, yj, mu = (
        params["bu"], params["bi"], params["pu"], params["qi"], params["yj"], params["mu"],
    )
    d = yj.shape[1]
    zeros_d = np.zeros(d)

    impl_sum = {u: impl_inv[u] * yj[items].sum(axis=0) for u, items in impl_N.items()}

    order = rng.permutation(len(train_ratings))
    sq_err = 0.0

    for idx in order:
        u, i, r = train_ratings[idx]
        implicit_factor = impl_sum.get(u, zeros_d)
        r_hat = float(np.clip(mu + bu[u] + bi[i] + qi[i].dot(pu[u] + implicit_factor), SVDPP_CLIP_MIN, SVDPP_CLIP_MAX))
        e = r - r_hat
        sq_err += e * e

        old_pu = pu[u].copy()
        old_qi = qi[i].copy()

        bu[u] += alpha * (e - lam * bu[u])
        bi[i] += alpha * (e - lam * bi[i])
        pu[u] += alpha * (e * old_qi - lam * pu[u])
        qi[i] += alpha * (e * (old_pu + implicit_factor) - lam * qi[i])
        N_u = impl_N.get(u)
        if N_u:
            yj[N_u] += alpha * (e * impl_inv[u] * old_qi - lam * yj[N_u])

    return params, sq_err / len(train_ratings)


def compute_rmse_mae(ratings, params, impl_N, impl_inv):
    bu, bi, pu, qi, yj, mu = (
        params["bu"], params["bi"], params["pu"], params["qi"], params["yj"], params["mu"],
    )
    d = yj.shape[1]

    us = np.fromiter((r[0] for r in ratings), dtype=np.int32, count=len(ratings))
    is_ = np.fromiter((r[1] for r in ratings), dtype=np.int32, count=len(ratings))
    rs = np.fromiter((r[2] for r in ratings), dtype=np.float64, count=len(ratings))

    user_eff = np.zeros((pu.shape[0], d))
    for u in np.unique(us):
        u = int(u)
        N_u = impl_N.get(u, [])
        impl_fac = impl_inv[u] * yj[N_u].sum(axis=0) if N_u else np.zeros(d)
        user_eff[u] = pu[u] + impl_fac

    r_hats = np.clip(
        mu + bu[us] + bi[is_] + (qi[is_] * user_eff[us]).sum(axis=1),
        SVDPP_CLIP_MIN,
        SVDPP_CLIP_MAX,
    )
    errors = rs - r_hats
    return math.sqrt(float((errors ** 2).mean())), float(np.abs(errors).mean())


def train_svdpp(
    train_ratings,
    val_ratings,
    params,
    hyperparams,
    impl_N,
    impl_inv,
    n_epochs: int,
    patience: int,
    shuffle_seed: int,
    verbose: bool = True,
):
    """Full training loop: SGD + LR decay + early stopping on val_ratings RMSE.

    val_ratings must be held-out rows the model never trains on (val_inner,
    carved from Train) -- NOT the final Test set, so model selection never
    touches Test.

    hyperparams is mutated in place (alpha decays each epoch) -- pass a copy
    if the caller needs the original values preserved.
    """
    rng = np.random.default_rng(shuffle_seed)
    best_rmse = float("inf")
    best_params = None
    no_improve = 0
    history = {"epoch": [], "train_rmse": [], "train_mae": [], "val_rmse": [], "val_mae": []}

    for epoch in range(1, n_epochs + 1):
        params, _ = train_one_epoch(train_ratings, params, hyperparams, impl_N, impl_inv, rng)
        train_rmse, train_mae = compute_rmse_mae(train_ratings, params, impl_N, impl_inv)
        val_rmse, val_mae = compute_rmse_mae(val_ratings, params, impl_N, impl_inv)
        hyperparams["alpha"] *= hyperparams["decay"]

        history["epoch"].append(epoch)
        history["train_rmse"].append(train_rmse)
        history["train_mae"].append(train_mae)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)

        if verbose:
            print(
                f"  Epoch {epoch:>3}/{n_epochs} | Train RMSE={train_rmse:.4f} | Val RMSE={val_rmse:.4f}"
            )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_params = copy.deepcopy(params)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}.")
                break

    return best_params, history


def recommend_topn(params, impl_N, impl_inv, user_id: int, n_items_max_id: int, exclude: set, n: int) -> list[tuple[int, float]]:
    """Rank the full item ID range [1, n_items_max_id] for user_id, excluding `exclude`.

    Candidate pool is intentionally the full ID range (not just items
    observed in train), asymmetric with DecayPop's train-observed-only
    pool -- confirmed by the audit as not a bug, documented here as a
    deliberate choice rather than something to unify.
    """
    all_items = [i for i in range(1, n_items_max_id + 1) if i not in exclude]
    scored = predict_batch(user_id, all_items, params, impl_N, impl_inv)
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def save_model(path: Path, params: dict, impl_N: dict, impl_inv: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"params": params, "impl_N": impl_N, "impl_inv": impl_inv}, f)


def load_model(path: Path) -> tuple[dict, dict, dict]:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob["params"], blob["impl_N"], blob["impl_inv"]
