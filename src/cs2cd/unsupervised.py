"""Unsupervised experiments on player-level features (no labels used to fit)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def _prep(players: pd.DataFrame, features: list[str]) -> np.ndarray:
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    return pipe.fit_transform(players[features].to_numpy(float))


def isolation_forest(players, features, seed=C.RANDOM_SEED) -> dict:
    X = _prep(players, features)
    iso = IsolationForest(n_estimators=500, random_state=seed).fit(X)
    score = -iso.score_samples(X)                  # higher = more anomalous
    y = players["label"].to_numpy()
    return {"method": "IsolationForest", "roc_auc": roc_auc_score(y, score),
            "pr_auc": average_precision_score(y, score), "score": score}


def kmeans(players, features, k_values=(2, 3, 4, 5), seed=C.RANDOM_SEED) -> list[dict]:
    X = _prep(players, features)
    y = players["label"].to_numpy()
    out = []
    for k in k_values:
        lab = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X)
        rates = pd.Series(y).groupby(lab).agg(["mean", "size"])
        best = int(rates["mean"].idxmax())
        flagged = lab == best
        tp = int((flagged & (y == 1)).sum())
        out.append({
            "method": f"KMeans(k={k})", "ari": adjusted_rand_score(y, lab),
            "cheater_cluster_size": int(flagged.sum()),
            "cheater_cluster_precision": tp / max(int(flagged.sum()), 1),
            "cheater_cluster_recall": tp / max(int(y.sum()), 1),
            "fpr": float(flagged[y == 0].mean()),
            "cluster_cheater_rates": rates["mean"].round(3).tolist(),
            "cluster_sizes": rates["size"].tolist(),
        })
    return out
