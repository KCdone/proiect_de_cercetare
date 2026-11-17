"""Integration test: synthetic match files -> features -> cross-validation.

A synthetic match is written in the CS2CD format (parquet + json). The
"cheater" snaps exactly onto the victim a few ticks before each kill, while
the honest player keeps an aim error of several degrees.
"""
import json

import numpy as np
import pandas as pd
import pytest

from cs2cd import config as C
from cs2cd.evaluate import cross_validate, make_models
from cs2cd.features import ALL_FEATURES, match_kill_table
from cs2cd.loader import load_match

N_TICKS = 3000


def _write_match(path, match_no: int, rng) -> None:
    players = {"Player_1": 2, "Player_2": 2, "Player_3": 3, "Player_4": 3}
    pos = {p: np.array([0.0, 0.0, 0.0]) if t == 2 else np.array([800.0, 0.0, 0.0])
           for p, t in players.items()}
    pos["Player_2"] = np.array([0.0, 300.0, 0.0])
    pos["Player_4"] = np.array([800.0, 300.0, 0.0])

    rows, deaths = [], []
    kills = [(400 + 250 * i, "Player_1" if i % 2 == 0 else "Player_2",
              "Player_3" if i % 3 else "Player_4") for i in range(10)]
    yaw = {p: rng.uniform(-180, 180) for p in players}
    pitch = {p: 0.0 for p in players}
    for t in range(1, N_TICKS + 1):
        for p, team in players.items():
            yaw[p] = (yaw[p] + rng.normal(0, 1.0) + 180) % 360 - 180
            for k, att, vic in kills:
                if p == att and k - 6 <= t <= k:
                    d = pos[vic] + [0, 0, C.TARGET_HEIGHT] - pos[att] - [0, 0, C.EYE_HEIGHT]
                    true_yaw = np.degrees(np.arctan2(d[1], d[0]))
                    true_pitch = -np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1])))
                    off = 0.0 if att == "Player_1" else 6.0
                    yaw[p], pitch[p] = true_yaw + off, true_pitch
            spotted = [a for k, a, v in kills if v == p and k - 20 <= t <= k]
            rows.append({"tick": t, "steamid": p, "pitch": pitch[p], "yaw": yaw[p],
                         "X": pos[p][0], "Y": pos[p][1], "Z": pos[p][2],
                         "is_alive": True, "team_num": float(team),
                         "approximate_spotted_by": spotted})
    for k, att, vic in kills:
        deaths.append({"tick": k, "attacker_steamid": att, "user_steamid": vic,
                       "weapon": "ak47", "headshot": att == "Player_1",
                       "distance": 800.0})
    pd.DataFrame(rows).to_parquet(path / f"{match_no}.parquet")
    (path / f"{match_no}.json").write_text(json.dumps({
        "player_death": deaths, "cheaters": [{"steamid": "Player_1"}],
        "CSstats_info": [{"map": "de_test"}]}))


@pytest.fixture(scope="module")
def synthetic_kills(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    rng = np.random.default_rng(0)
    tables = []
    for i in range(10):
        _write_match(d, i, rng)
        tables.append(match_kill_table(load_match(d / f"{i}.parquet")))
    return pd.concat(tables, ignore_index=True)


def test_loader_and_features_produce_one_row_per_kill(synthetic_kills):
    assert len(synthetic_kills) == 10 * 10
    assert set(ALL_FEATURES) <= set(synthetic_kills.columns)
    assert synthetic_kills["label"].sum() == 10 * 5      # Player_1's kills


def test_cheater_aim_error_is_smaller(synthetic_kills):
    g = synthetic_kills.groupby("label")["err_at_kill"].mean()
    assert g[1] < 0.5 < g[0]


def test_visibility_feature_is_found(synthetic_kills):
    assert synthetic_kills["time_to_kill_ticks"].notna().all()


def test_cross_validation_end_to_end(synthetic_kills):
    res = cross_validate(synthetic_kills, ALL_FEATURES, make_models()["LogReg"])
    assert len(res.folds) == C.N_OUTER_FOLDS
    assert res.folds["kill_roc_auc"].mean() > 0.9
    # no match may appear in more than one test fold
    assert res.kill_scores.groupby("match_id")["fold"].nunique().max() == 1
