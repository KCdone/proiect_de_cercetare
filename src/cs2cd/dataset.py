"""Builds the kill-level feature table for a set of downloaded matches."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from . import config as C
from .features import match_kill_table
from .loader import load_match


def _one(parquet: Path) -> pd.DataFrame:
    try:
        m = load_match(parquet)
        t = match_kill_table(m)
        if not t.empty:
            t["map"] = m.meta.get("map", "")
            t["avg_rank"] = m.meta.get("avg_rank", "")
        return t
    except Exception as e:                              # corrupted file, etc.
        print(f"[skip] {parquet.name}: {e}")
        return pd.DataFrame()


def build(raw_dir: Path, out: Path, workers: int = 2) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.parquet"), key=lambda p: int(p.stem))
    with ProcessPoolExecutor(workers) as ex:
        tables = list(ex.map(_one, files))
    df = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"{len(files)} matches -> {len(df)} kills, "
          f"{df.groupby(['match_id', 'player']).ngroups} player-matches -> {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=C.RAW_DIR / C.HF_SUBSET)
    ap.add_argument("--out", type=Path, default=C.FEATURES_DIR / "kills.csv")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    build(a.raw, a.out, a.workers)
