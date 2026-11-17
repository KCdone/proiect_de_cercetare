"""Unit tests for thresholds, aggregation and metrics."""
import numpy as np
import pandas as pd
import pytest

from cs2cd.evaluate import aggregate_players, binary_metrics, threshold_at_fpr


def test_threshold_respects_target_fpr():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(200)].astype(int)
    s = np.r_[rng.normal(0, 1, 1000), rng.normal(2, 1, 200)]
    t = threshold_at_fpr(y, s, 0.05)
    fpr = (s[y == 0] >= t).mean()
    assert fpr <= 0.05
    assert fpr > 0.04            # and it is not needlessly strict


def test_threshold_zero_fpr_is_above_every_negative():
    y = np.array([0, 0, 1])
    s = np.array([0.2, 0.7, 0.9])
    assert threshold_at_fpr(y, s, 0.0) > 0.7


def test_aggregate_players_drops_players_with_few_kills():
    kills = pd.DataFrame({
        "match_id": [1, 1, 1, 1, 2],
        "player": ["a", "a", "a", "b", "a"],
        "label": [1, 1, 1, 0, 0],
        "score": [0.9, 0.8, 0.1, 0.3, 0.2],
    })
    p = aggregate_players(kills, min_kills=3)
    assert len(p) == 1
    row = p.iloc[0]
    assert row["score_mean"] == pytest.approx(0.6)
    assert row["score_top3"] == pytest.approx(0.6)


def test_binary_metrics_perfect_classifier():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert m["roc_auc"] == 1.0 and m["f1"] == 1.0 and m["fpr"] == 0.0
