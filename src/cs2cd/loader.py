"""Loading one CS2CD match (tick table + event file)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

TICK_COLUMNS = ["tick", "steamid", "pitch", "yaw", "X", "Y", "Z",
                "is_alive", "team_num", "approximate_spotted_by"]


@dataclass
class Match:
    match_id: str
    ticks: pd.DataFrame            # one row per (tick, player)
    deaths: pd.DataFrame           # player_death events
    cheaters: set[str] = field(default_factory=set)
    meta: dict = field(default_factory=dict)

    @property
    def players(self) -> list[str]:
        return sorted(p for p in self.ticks["steamid"].dropna().unique() if p)


def load_match(parquet_path: str | Path, json_path: str | Path | None = None,
               match_id: str | None = None) -> Match:
    parquet_path = Path(parquet_path)
    json_path = Path(json_path) if json_path else parquet_path.with_suffix(".json")
    match_id = match_id or parquet_path.stem

    available = pq.ParquetFile(parquet_path).schema_arrow.names
    cols = [c for c in TICK_COLUMNS if c in available]
    ticks = pq.read_table(parquet_path, columns=cols).to_pandas()
    ticks = ticks.dropna(subset=["steamid", "pitch", "yaw"])

    with open(json_path, encoding="utf-8") as f:
        events = json.load(f)

    deaths = pd.DataFrame(events.get("player_death", []))
    cheaters = {c["steamid"] for c in events.get("cheaters", []) if c.get("steamid")}
    meta = (events.get("CSstats_info") or [{}])[0]
    return Match(match_id, ticks, deaths, cheaters, meta)
