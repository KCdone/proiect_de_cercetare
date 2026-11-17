"""Supervised experiments: kill-level models + player-level aggregation.

Protocol
--------
* Outer loop: StratifiedGroupKFold over kills, grouped by match, so that no
  match (and therefore no player-match) is split between train and test.
* For each outer fold an inner StratifiedGroupKFold on the training part gives
  out-of-fold scores, used ONLY to pick the decision thresholds so that the
  false positive rate on the training data is <= TARGET_FPR.
* Kill scores are aggregated into one score per (match, player).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config as C


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def make_models(seed: int = C.RANDOM_SEED) -> dict:
    imp = lambda: SimpleImputer(strategy="median", add_indicator=False)  # noqa: E731
    return {
        "LogReg": make_pipeline(imp(), StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced")),
        "RandomForest": make_pipeline(imp(), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   class_weight="balanced", random_state=seed, verbose=-1),
        "SVM": make_pipeline(imp(), StandardScaler(),
                             SVC(C=1.0, kernel="rbf", probability=True,
                                 class_weight="balanced", random_state=seed)),
        "MLP": make_pipeline(imp(), StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3,
                                           early_stopping=True, max_iter=500,
                                           random_state=seed)),
    }


# ----------------------------------------------------------------------------
# Thresholds, aggregation, metrics (unit tested)
# ----------------------------------------------------------------------------
def threshold_at_fpr(y: np.ndarray, s: np.ndarray, target_fpr: float = C.TARGET_FPR) -> float:
    """Smallest threshold t such that FPR(score >= t) <= target_fpr."""
    neg = np.sort(np.asarray(s)[np.asarray(y) == 0])
    if len(neg) == 0:
        return 0.5
    k = int(np.floor(target_fpr * len(neg)))     # negatives allowed above t
    if k == 0:
        return float(np.nextafter(neg[-1], np.inf))
    return float(np.nextafter(neg[-k - 1], np.inf))


def aggregate_players(kills: pd.DataFrame, score_col: str = "score",
                      min_kills: int = C.MIN_KILLS_PER_PLAYER) -> pd.DataFrame:
    """One row per (match, player): mean / top-3 mean of the kill scores."""
    g = kills.groupby(["match_id", "player"])
    out = g.agg(label=("label", "max"), n_kills=("label", "size"),
                score_mean=(score_col, "mean"),
                score_top3=(score_col, lambda x: np.sort(x.to_numpy())[-3:].mean()))
    return out[out["n_kills"] >= min_kills].reset_index()


def binary_metrics(y, s, thr) -> dict:
    y, s = np.asarray(y), np.asarray(s)
    pred = (s >= thr).astype(int)
    neg = y == 0
    return {
        "roc_auc": roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan,
        "pr_auc": average_precision_score(y, s) if y.sum() > 0 else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": float(pred[neg].mean()) if neg.any() else np.nan,
        "n": int(len(y)), "n_pos": int(y.sum()),
    }


# ----------------------------------------------------------------------------
# Cross-validation
# ----------------------------------------------------------------------------
@dataclass
class CVResult:
    kill_scores: pd.DataFrame       # out-of-fold kill scores
    player_scores: pd.DataFrame     # out-of-fold player scores + fold thresholds
    folds: pd.DataFrame             # per-fold metrics


def _oof(model, X, y, groups, n_splits, seed) -> np.ndarray:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    out = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        m = clone(model).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


def cross_validate(kills: pd.DataFrame, features: list[str], model,
                   agg: str = "score_mean", seed: int = C.RANDOM_SEED) -> CVResult:
    X = kills[features].to_numpy(float)
    y = kills["label"].to_numpy(int)
    groups = kills["match_id"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=C.N_OUTER_FOLDS, shuffle=True, random_state=seed)

    kill_parts, player_parts, rows = [], [], []
    for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        # thresholds from inner out-of-fold scores on the training part only
        inner = _oof(model, X[tr], y[tr], groups[tr], C.N_INNER_FOLDS, seed)
        k_thr = threshold_at_fpr(y[tr], inner)
        tr_df = kills.iloc[tr][["match_id", "player", "label"]].assign(score=inner)
        tr_players = aggregate_players(tr_df)
        p_thr = threshold_at_fpr(tr_players["label"], tr_players[agg])

        m = clone(model).fit(X[tr], y[tr])
        s = m.predict_proba(X[te])[:, 1]
        te_df = kills.iloc[te][["match_id", "player", "label"]].assign(score=s, fold=fold)
        te_players = aggregate_players(te_df).assign(fold=fold, thr=p_thr)

        km = binary_metrics(y[te], s, k_thr)
        pm = binary_metrics(te_players["label"], te_players[agg], p_thr)
        rows.append({"fold": fold, **{f"kill_{k}": v for k, v in km.items()},
                     **{f"player_{k}": v for k, v in pm.items()}})
        kill_parts.append(te_df)
        player_parts.append(te_players)

    return CVResult(pd.concat(kill_parts), pd.concat(player_parts), pd.DataFrame(rows))


def summarize(res: CVResult) -> dict:
    """Mean and std of every per-fold metric."""
    f = res.folds.drop(columns="fold")
    out = {}
    for c in f.columns:
        out[c] = f[c].mean()
        out[c + "_std"] = f[c].std()
    return out


# ----------------------------------------------------------------------------
# Player-level models (baseline and aggregated-feature model)
# ----------------------------------------------------------------------------
def player_feature_table(kills: pd.DataFrame, features: list[str],
                         min_kills: int = C.MIN_KILLS_PER_PLAYER) -> pd.DataFrame:
    """Mean and max of each kill feature per (match, player)."""
    g = kills.groupby(["match_id", "player"])
    mean = g[features].mean().add_suffix("_mean")
    mx = g[features].max().add_suffix("_max")
    base = g.agg(label=("label", "max"), n_kills=("label", "size"),
                 hs_rate=("headshot", "mean"))
    t = pd.concat([base, mean, mx], axis=1).reset_index()
    return t[t["n_kills"] >= min_kills].reset_index(drop=True)


def cross_validate_players(players: pd.DataFrame, features: list[str], model,
                           seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    X = players[features].to_numpy(float)
    y = players["label"].to_numpy(int)
    groups = players["match_id"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=C.N_OUTER_FOLDS, shuffle=True, random_state=seed)
    rows, parts = [], []
    for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        inner = _oof(model, X[tr], y[tr], groups[tr], C.N_INNER_FOLDS, seed)
        thr = threshold_at_fpr(y[tr], inner)
        s = clone(model).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        pm = binary_metrics(y[te], s, thr)
        rows.append({"fold": fold, **{f"player_{k}": v for k, v in pm.items()}})
        parts.append(players.iloc[te][["match_id", "player", "label"]].assign(score=s, fold=fold))
    return CVResult(pd.DataFrame(), pd.concat(parts), pd.DataFrame(rows))
