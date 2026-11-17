"""Downloads a subset of CS2CD matches from HuggingFace."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from . import config as C


def download(n: int, subset: str = C.HF_SUBSET, out: Path = C.RAW_DIR) -> None:
    for i in range(n):
        for ext in ("json", "parquet"):
            target = out / subset / f"{i}.{ext}"
            if target.exists():
                continue
            hf_hub_download(C.HF_REPO_ID, f"{subset}/{i}.{ext}", repo_type="dataset",
                            local_dir=out)
        print(f"match {i} ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=100, help="number of matches")
    download(ap.parse_args().n)
