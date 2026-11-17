# Server-side behavioural cheat detection in Counter-Strike 2

Code for the case study of the bachelor thesis *Anti-cheat systems in online
games* (UBB Cluj, FMI). A detector that works only on server-side replay data
(view angles, positions, kill events) is trained on the public
[CS2CD](https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection)
dataset (CC-BY-4.0) and decides, per match, whether a player was cheating.

## Pipeline

```
CS2CD match (.parquet ticks + .json events)
   └─ loader.py        load one match, labels from the "cheaters" field
   └─ features.py      window of 64 + 16 ticks around every kill → 24 features
   └─ dataset.py       kill-level table for all matches  (data/features/kills.csv)
   └─ evaluate.py      kill-level models, match-grouped CV, FPR ≤ 5 % threshold,
                       aggregation per (match, player)
   └─ unsupervised.py  Isolation Forest and k-means on player features
   └─ run_experiments.py   experiments E1–E6 → results/
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=src                                   # Windows: set PYTHONPATH=src

python -m cs2cd.download -n 100        # ~4.7 GB, 100 matches with a cheater
python -m cs2cd.dataset                # ~30 s  -> data/features/kills.csv
python -m cs2cd.run_experiments        # ~7 min -> results/
pytest -q                              # unit + integration tests
```

With Docker:

```bash
docker build -t cs2cd .
docker run --rm cs2cd                  # runs the tests
docker compose up                      # download + features + experiments
```

## Experiments

| ID | Question | Output |
|----|----------|--------|
| E1 | Which model works best (LogReg, RF, LightGBM, SVM, MLP)? | `E1_models_*.csv`, `figures/roc_player_level.png` |
| E2 | Which feature groups matter (angles / target / context)? | `E2_feature_ablation.csv` |
| E3 | Per-kill vs per-player decision; classic statistical baseline | `E3_decision_level_and_baselines.csv` |
| E4 | Can unsupervised methods separate cheaters? | `E4_unsupervised.csv` |
| E5 | Which features drive the decision (permutation, SHAP)? | `E5_*.csv`, `figures/` |
| E6 | Robustness without SSG08 kills / visibility features | `E6_robustness.csv` |

Main result (100 matches, 729 player-matches with ≥ 3 kills, 5-fold CV
grouped by match): LightGBM, player level, ROC AUC 0.950, recall 0.81 at an
FPR of 0.034. See `SUMMARY.md`.

## Repository layout

```
src/cs2cd/           source code
tests/               pytest unit and integration tests (synthetic matches)
results/             metrics (CSV) and figures of the last run
Dockerfile, docker-compose.yml   reproducible environment
.github/workflows/ci.yml         CI: lint, tests, Docker build
```

The raw data is not committed (`data/` is in `.gitignore`).
