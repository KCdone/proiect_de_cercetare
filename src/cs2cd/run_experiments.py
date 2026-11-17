"""Runs experiments E1-E5 of the case study and writes results/ + figures."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from . import config as C  # noqa: E402
from .evaluate import (cross_validate, cross_validate_players, make_models,  # noqa: E402
                       player_feature_table, summarize)
from .features import ANGLE_FEATURES, CONTEXT_FEATURES, TARGET_FEATURES  # noqa: E402
from .unsupervised import isolation_forest, kmeans  # noqa: E402

FEATURE_SETS = {
    "A (angles only)": ANGLE_FEATURES,
    "A+T (angles + target)": ANGLE_FEATURES + TARGET_FEATURES,
    "A+T+C (all)": ANGLE_FEATURES + TARGET_FEATURES + CONTEXT_FEATURES,
}
FULL = FEATURE_SETS["A+T+C (all)"]
KEEP = ["roc_auc", "pr_auc", "precision", "recall", "f1", "fpr"]


def _row(name: str, s: dict, level: str) -> dict:
    r = {"setting": name}
    for k in KEEP:
        r[k] = s.get(f"{level}_{k}")
        r[k + "_std"] = s.get(f"{level}_{k}_std")
    return r


def main(kills_csv: Path, out: Path, models: list[str] | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    kills = pd.read_csv(kills_csv)
    all_models = make_models()
    models = models or list(all_models)
    log = {"n_kills": len(kills), "n_matches": int(kills.match_id.nunique()),
           "kill_cheater_share": float(kills.label.mean())}
    t0 = time.time()

    # ---- E1: model comparison on all features ------------------------------
    e1_k, e1_p, roc_data = [], [], {}
    for name in models:
        res = cross_validate(kills, FULL, all_models[name])
        s = summarize(res)
        e1_k.append(_row(name, s, "kill"))
        e1_p.append(_row(name, s, "player"))
        ps = res.player_scores
        roc_data[name] = roc_curve(ps["label"], ps["score_mean"])
        res.player_scores.to_csv(out / f"oof_players_{name}.csv", index=False)
        print(f"E1 {name}: kill AUC {s['kill_roc_auc']:.3f}  player AUC "
              f"{s['player_roc_auc']:.3f}  ({time.time() - t0:.0f}s)")
        if name == "LightGBM":
            log["players_evaluated"] = int(len(ps))
            log["player_cheater_share"] = float(ps["label"].mean())
    pd.DataFrame(e1_k).to_csv(out / "E1_models_kill_level.csv", index=False)
    pd.DataFrame(e1_p).to_csv(out / "E1_models_player_level.csv", index=False)

    plt.figure(figsize=(5.5, 5))
    for name, (fpr, tpr, _) in roc_data.items():
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "k:", lw=0.8)
    plt.axvline(C.TARGET_FPR, color="grey", ls="--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Player-level ROC (out-of-fold)"); plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir / "roc_player_level.png", dpi=160); plt.close()

    # ---- E2: feature-group ablation ----------------------------------------
    e2 = []
    for fs_name, fs in FEATURE_SETS.items():
        for name in ["RandomForest", "LightGBM"]:
            s = summarize(cross_validate(kills, fs, all_models[name]))
            e2.append({"features": fs_name, "model": name,
                       "kill_roc_auc": s["kill_roc_auc"], "kill_roc_auc_std": s["kill_roc_auc_std"],
                       "player_roc_auc": s["player_roc_auc"],
                       "player_roc_auc_std": s["player_roc_auc_std"],
                       "player_f1": s["player_f1"], "player_fpr": s["player_fpr"]})
            print(f"E2 {fs_name} {name}: player AUC {s['player_roc_auc']:.3f}")
    pd.DataFrame(e2).to_csv(out / "E2_feature_ablation.csv", index=False)

    # ---- E3: decision level and baselines ----------------------------------
    lgbm = all_models["LightGBM"]
    players = player_feature_table(kills, FULL)
    e3 = []
    s = summarize(cross_validate(kills, FULL, lgbm, agg="score_mean"))
    e3.append({"setting": "Per-kill decision (LightGBM)", **{k: s[f"kill_{k}"] for k in KEEP}})
    e3.append({"setting": "Player: mean of kill scores", **{k: s[f"player_{k}"] for k in KEEP}})
    s = summarize(cross_validate(kills, FULL, lgbm, agg="score_top3"))
    e3.append({"setting": "Player: mean of top-3 kill scores", **{k: s[f"player_{k}"] for k in KEEP}})
    pf = [c for c in players.columns if c.endswith("_mean") or c.endswith("_max")]
    s = summarize(cross_validate_players(players, pf, lgbm))
    e3.append({"setting": "Player: LightGBM on aggregated features", **{k: s[f"player_{k}"] for k in KEEP}})
    base = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced"))
    s = summarize(cross_validate_players(players, ["hs_rate", "n_kills"], base))
    e3.append({"setting": "Baseline: headshot rate + kill count (LogReg)",
               **{k: s[f"player_{k}"] for k in KEEP}})
    pd.DataFrame(e3).to_csv(out / "E3_decision_level_and_baselines.csv", index=False)
    print("E3 done")

    # ---- E4: unsupervised --------------------------------------------------
    uf = [f + "_mean" for f in ANGLE_FEATURES + TARGET_FEATURES]
    iso = isolation_forest(players, uf)
    km = kmeans(players, uf)
    e4 = [{"method": iso["method"], "roc_auc": iso["roc_auc"], "pr_auc": iso["pr_auc"]}] + km
    pd.DataFrame(e4).to_csv(out / "E4_unsupervised.csv", index=False)
    print("E4 done")

    # ---- E5: explainability ------------------------------------------------
    X, y, g = kills[FULL].to_numpy(float), kills["label"].to_numpy(), kills["match_id"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=C.N_OUTER_FOLDS, shuffle=True, random_state=C.RANDOM_SEED)
    imps = []
    for tr, te in cv.split(X, y, g):
        m = make_models()["LightGBM"].fit(X[tr], y[tr])
        r = permutation_importance(m, X[te], y[te], scoring="roc_auc", n_repeats=5,
                                   random_state=C.RANDOM_SEED, n_jobs=1)
        imps.append(r.importances_mean)
    imp = pd.DataFrame({"feature": FULL, "auc_drop_mean": np.mean(imps, 0),
                        "auc_drop_std": np.std(imps, 0)}).sort_values("auc_drop_mean", ascending=False)
    imp.to_csv(out / "E5_permutation_importance.csv", index=False)
    top = imp.head(12)[::-1]
    plt.figure(figsize=(6, 4.5))
    plt.barh(top["feature"], top["auc_drop_mean"], xerr=top["auc_drop_std"], color="#4C72B0")
    plt.xlabel("Drop in kill-level ROC AUC when permuted")
    plt.title("Permutation importance (LightGBM)"); plt.tight_layout()
    plt.savefig(fig_dir / "permutation_importance.png", dpi=160); plt.close()

    try:
        import shap
        m = make_models()["LightGBM"].fit(X, y)
        rng = np.random.default_rng(C.RANDOM_SEED)
        idx = rng.choice(len(X), size=min(2000, len(X)), replace=False)
        sv = shap.TreeExplainer(m).shap_values(X[idx])
        sv = sv[1] if isinstance(sv, list) else sv
        shap_imp = pd.DataFrame({"feature": FULL, "mean_abs_shap": np.abs(sv).mean(0)}) \
            .sort_values("mean_abs_shap", ascending=False)
        shap_imp.to_csv(out / "E5_shap_importance.csv", index=False)
        shap.summary_plot(sv, X[idx], feature_names=FULL, show=False, max_display=12)
        plt.tight_layout(); plt.savefig(fig_dir / "shap_summary.png", dpi=160); plt.close()
    except Exception as e:  # shap is optional
        print("SHAP skipped:", e)

    # distribution figure for the two most informative target features
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for a, f, lim in [(ax[0], "err_mean_last16", 20), (ax[1], "ang_acc_abs_max", 15)]:
        for lab, name in [(0, "not cheater"), (1, "cheater")]:
            v = kills.loc[kills.label == lab, f].clip(upper=lim)
            a.hist(v, bins=40, alpha=0.55, density=True, label=name)
        a.set_title(f); a.legend()
    plt.tight_layout(); plt.savefig(fig_dir / "feature_distributions.png", dpi=160); plt.close()

    log["runtime_s"] = round(time.time() - t0, 1)
    (out / "run_info.json").write_text(json.dumps(log, indent=2))
    print("All experiments done in", log["runtime_s"], "s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kills", type=Path, default=C.FEATURES_DIR / "kills.csv")
    ap.add_argument("--out", type=Path, default=C.RESULTS_DIR)
    ap.add_argument("--models", nargs="*", default=None)
    a = ap.parse_args()
    main(a.kills, a.out, a.models)
