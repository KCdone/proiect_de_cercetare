"""Kill-window feature extraction.

For every kill made with an aiming weapon, the attacker's view angles are
taken in the window [k - WINDOW_BEFORE_TICKS, k + WINDOW_AFTER_TICKS]
around the kill tick k. Three feature groups are computed:

* A  - angle dynamics (only pitch/yaw of the attacker; game-independent)
* T  - target-relative aim (needs attacker/victim positions and visibility)
* C  - classic context statistics of the kill (headshot, distance)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .loader import Match

EPS = 1e-6

ANGLE_FEATURES = [
    "ang_speed_mean", "ang_speed_max", "ang_speed_std",
    "ang_acc_abs_mean", "ang_acc_abs_max", "ang_acc_std",
    "jerk_abs_mean", "jerk_abs_max",
    "snap_ratio", "peak_to_kill_ticks", "still_ticks_before_kill",
    "total_angle_pre", "post_kill_speed_mean",
]
TARGET_FEATURES = [
    "err_at_kill", "err_mean_last16", "err_min_window", "err_at_start",
    "err_reduction", "on_target_ticks", "time_to_kill_ticks", "err_at_spot",
    "unseen_kill",
]
CONTEXT_FEATURES = ["headshot", "distance"]
ALL_FEATURES = ANGLE_FEATURES + TARGET_FEATURES + CONTEXT_FEATURES


# ----------------------------------------------------------------------------
# Elementary maths (unit tested)
# ----------------------------------------------------------------------------
def wrap_deg(d: np.ndarray) -> np.ndarray:
    """Wrap angle differences into [-180, 180)."""
    return (np.asarray(d, dtype=float) + 180.0) % 360.0 - 180.0


def angular_speed(pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Angular speed in degrees per tick between consecutive ticks."""
    dp = np.diff(np.asarray(pitch, dtype=float))
    dy = wrap_deg(np.diff(np.asarray(yaw, dtype=float)))
    return np.sqrt(dp ** 2 + dy ** 2)


def view_vector(pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """Unit view direction. In Source 2 a positive pitch looks down."""
    p = np.radians(np.asarray(pitch, dtype=float))
    y = np.radians(np.asarray(yaw, dtype=float))
    return np.stack([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), -np.sin(p)], axis=-1)


def aim_error(pitch, yaw, eye_pos, target_pos) -> np.ndarray:
    """Angle (degrees) between the view direction and the eye->target line."""
    v = view_vector(pitch, yaw)
    u = np.asarray(target_pos, dtype=float) - np.asarray(eye_pos, dtype=float)
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + EPS)
    cos = np.clip(np.sum(v * u, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


# ----------------------------------------------------------------------------
# Per-match extraction
# ----------------------------------------------------------------------------
class _PlayerTrack:
    """Dense per-tick arrays of one player (missing ticks forward-filled)."""

    def __init__(self, df: pd.DataFrame, t0: int, t1: int):
        df = df.drop_duplicates("tick").set_index("tick").reindex(range(t0, t1 + 1))
        num = df[["pitch", "yaw", "X", "Y", "Z", "team_num"]].ffill().bfill()
        self.t0 = t0
        self.pitch = num["pitch"].to_numpy(float)
        self.yaw = num["yaw"].to_numpy(float)
        self.pos = num[["X", "Y", "Z"]].to_numpy(float)
        self.team = num["team_num"].to_numpy(float)
        if "approximate_spotted_by" in df:
            self.spotted = df["approximate_spotted_by"].to_numpy(object)
        else:
            self.spotted = None

    def sl(self, a: int, b: int) -> slice:
        return slice(a - self.t0, b - self.t0 + 1)


def _spotted_by(entry, who: str) -> bool:
    if entry is None or (isinstance(entry, float) and np.isnan(entry)):
        return False
    try:
        return who in list(entry)
    except TypeError:
        return False


def kill_features(att: _PlayerTrack, vic: _PlayerTrack, k: int,
                  attacker_id: str) -> dict:
    a, b = k - C.WINDOW_BEFORE_TICKS, k + C.WINDOW_AFTER_TICKS
    s = att.sl(a, b)
    pitch, yaw = att.pitch[s], att.yaw[s]
    n_pre = C.WINDOW_BEFORE_TICKS            # index of the kill tick in window

    w = angular_speed(pitch, yaw)            # len = window - 1
    w_pre, w_post = w[:n_pre], w[n_pre:]
    acc = np.diff(w)
    jerk = np.diff(acc)
    acc_pre, jerk_pre = acc[: n_pre - 1], jerk[: n_pre - 2]

    late = w_pre[-16:]
    early = w_pre[:-16]
    f = {
        "ang_speed_mean": w_pre.mean(),
        "ang_speed_max": w_pre.max(),
        "ang_speed_std": w_pre.std(),
        "ang_acc_abs_mean": np.abs(acc_pre).mean(),
        "ang_acc_abs_max": np.abs(acc_pre).max(),
        "ang_acc_std": acc_pre.std(),
        "jerk_abs_mean": np.abs(jerk_pre).mean(),
        "jerk_abs_max": np.abs(jerk_pre).max(),
        "snap_ratio": late.max() / (early.mean() + 0.05),
        "peak_to_kill_ticks": float(n_pre - 1 - int(np.argmax(w_pre))),
        "still_ticks_before_kill": float((w_pre[-8:] < 0.05).sum()),
        "total_angle_pre": w_pre.sum(),
        "post_kill_speed_mean": w_post.mean() if len(w_post) else 0.0,
    }

    # --- target-relative aim -------------------------------------------------
    eye = att.pos[s] + np.array([0, 0, C.EYE_HEIGHT])
    tgt = vic.pos[vic.sl(a, b)] + np.array([0, 0, C.TARGET_HEIGHT])
    err = aim_error(pitch, yaw, eye, tgt)
    err_pre = err[: n_pre + 1]
    f.update({
        "err_at_kill": err_pre[-1],
        "err_mean_last16": err_pre[-16:].mean(),
        "err_min_window": err_pre.min(),
        "err_at_start": err_pre[0],
        "err_reduction": err_pre[0] - err_pre[-1],
        "on_target_ticks": float((err_pre < 3.0).sum()),
    })

    # visibility: first tick of the continuous "spotted" streak ending at k
    tts, err_spot = np.nan, np.nan
    if vic.spotted is not None:
        lo = max(vic.t0, k - C.SPOT_LOOKBACK_TICKS)
        first = None
        for t in range(k, lo - 1, -1):
            if _spotted_by(vic.spotted[t - vic.t0], attacker_id):
                first = t
            elif first is not None or t < k - 8:
                break
        if first is not None:
            tts = float(k - first)
            e = aim_error(att.pitch[first - att.t0], att.yaw[first - att.t0],
                          att.pos[first - att.t0] + [0, 0, C.EYE_HEIGHT],
                          vic.pos[first - vic.t0] + [0, 0, C.TARGET_HEIGHT])
            err_spot = float(e)
    f["time_to_kill_ticks"] = tts
    f["err_at_spot"] = err_spot
    f["unseen_kill"] = float(np.isnan(tts))
    return f


def match_kill_table(m: Match) -> pd.DataFrame:
    """One row per valid kill of the match, with features and labels."""
    if m.deaths.empty:
        return pd.DataFrame()
    t0, t1 = int(m.ticks["tick"].min()), int(m.ticks["tick"].max())
    tracks = {p: _PlayerTrack(g, t0, t1) for p, g in m.ticks.groupby("steamid") if p}

    rows = []
    for d in m.deaths.itertuples(index=False):
        att, vic = getattr(d, "attacker_steamid", ""), getattr(d, "user_steamid", "")
        weapon = str(getattr(d, "weapon", "")).lower()
        k = int(d.tick)
        if not att or not vic or att == vic or weapon in C.NON_AIM_WEAPONS:
            continue
        if att not in tracks or vic not in tracks:
            continue
        if k - C.WINDOW_BEFORE_TICKS < t0 or k + C.WINDOW_AFTER_TICKS > t1:
            continue
        A, V = tracks[att], tracks[vic]
        if A.team[k - t0] == V.team[k - t0]:          # skip team kills
            continue
        f = kill_features(A, V, k, att)
        f.update({
            "headshot": float(bool(getattr(d, "headshot", False))),
            "distance": float(getattr(d, "distance", np.nan)),
            "match_id": m.match_id, "player": att, "victim": vic,
            "tick": k, "weapon": weapon,
            "label": int(att in m.cheaters),
        })
        rows.append(f)
    return pd.DataFrame(rows)
