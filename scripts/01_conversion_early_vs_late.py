"""E-commerce conversion: early-session foresight vs end-of-session tautology.

Data: Data Mining Cup 2013 ("prediction of orders") - anonymised real data from
a GENERIC online shop (the task PDF says "online shop", not fashion). Target:
per-session `order` (y/n). ~50,000 sessions / ~429,000 transaction (click) rows.

Honest spine (no fabricated euro figures - there is no clean in-data margin/AOV
anchor):
  * The headline 67.6% "order rate" is the curated COMPETITION population, not a
    real funnel conversion rate (real funnels are low single digits). Labelled as
    such everywhere.
  * Validation splits by SESSION (sessionNo), never by transaction row - a
    session-level label with many rows per session would leak on a row split.
  * "Early" model uses only each session's FIRST transaction (what you know after
    the first interaction). "Full" model uses the LAST transaction (final basket
    state). The gap between the two AUCs is real foresight vs end-state tautology:
    basket features (bCount, bSumPrice, bStep) trivially encode the outcome at the
    end of the session.

Outputs: results/metrics.json, figures/01_early_vs_late.png.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

REPO = Path(__file__).resolve().parents[1]
FIG, RES = REPO / "figures", REPO / "results"
FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)

NUM = ["startHour", "startWeekday", "duration", "cCount", "cMinPrice", "cMaxPrice",
       "cSumPrice", "bCount", "bMinPrice", "bMaxPrice", "bSumPrice", "bStep",
       "maxVal", "customerScore", "accountLifetime", "age"]
CAT = ["onlineStatus", "availability", "address", "payments", "lastOrder"]
BASKET = ["bCount", "bMinPrice", "bMaxPrice", "bSumPrice", "bStep"]  # the leak-prone block
FEATS = NUM + CAT


def find_data() -> str:
    import os
    env = os.environ.get("DMC2013_DATA")
    if env and Path(env).is_file():
        return env
    hits = sorted(glob.glob(str(REPO / "data/raw/*transact_train*"))
                  + glob.glob(str(Path.home() / ".cache/kagglehub/**/transact_train.txt"), recursive=True))
    if hits:
        return hits[0]
    import kagglehub
    base = kagglehub.dataset_download("oscarm524/prediction-of-orders")
    return sorted(glob.glob(str(Path(base) / "**/transact_train.txt"), recursive=True))[0]


def session_frame(df: pd.DataFrame, which: str) -> pd.DataFrame:
    """One row per session: 'first' = earliest transaction, 'last' = final state."""
    g = df.sort_values(["sessionNo", "cCount"]).groupby("sessionNo", sort=True)
    rows = g.first() if which == "first" else g.last()
    rows["order"] = g["order"].last()
    return rows.reset_index()


def fit_auc(tr, te, feats):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    Xtr, Xte = tr[feats].copy(), te[feats].copy()
    cats = [c for c in CAT if c in feats]
    if cats:
        Xtr[cats] = enc.fit_transform(tr[cats].astype(str))
        Xte[cats] = enc.transform(te[cats].astype(str))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_depth=8, random_state=42)
    clf.fit(Xtr, tr["y"].values)
    p = clf.predict_proba(Xte)[:, 1]
    return clf, p, float(roc_auc_score(te["y"].values, p))


def main() -> None:
    df = pd.read_csv(find_data(), sep="|", na_values="?")
    # sanity: order is a per-session label
    per_sess_unique = df.groupby("sessionNo")["order"].nunique()
    assert per_sess_unique.max() == 1, "order is not constant within a session!"

    first = session_frame(df, "first")
    last = session_frame(df, "last")
    for d in (first, last):
        d["y"] = (d["order"] == "y").astype(int)

    n_sessions = first["sessionNo"].nunique()
    base = float(first["y"].mean())

    # GROUP split by session: oldest 80% of sessionNo -> train, newest 20% -> test
    cut = first["sessionNo"].quantile(0.80)
    def split(d):
        return d[d["sessionNo"] <= cut], d[d["sessionNo"] > cut]
    f_tr, f_te = split(first)
    l_tr, l_te = split(last)

    _, _, auc_early = fit_auc(f_tr, f_te, FEATS)
    _, p_full, auc_full = fit_auc(l_tr, l_te, FEATS)
    # full model WITHOUT the basket block -> isolates the basket-leakage contribution
    _, _, auc_full_nobasket = fit_auc(l_tr, l_te, [c for c in FEATS if c not in BASKET])

    yte = l_te["y"].values
    order = np.argsort(-p_full)
    lift1 = float(yte[order[: len(order) // 10]].mean() / base)
    brier = None
    fp, mp = calibration_curve(yte, p_full, n_bins=10, strategy="quantile")

    m = {
        "dataset": "DMC 2013 (prediction of orders) - GENERIC online shop, not fashion",
        "n_sessions": int(n_sessions),
        "population_order_rate": round(base, 4),
        "population_note": "curated competition population, NOT a real funnel conversion rate (real funnels are low single digits)",
        "split": "by sessionNo, oldest 80% train / newest 20% test (group split, no row leakage)",
        "auc_early_first_transaction": round(auc_early, 4),
        "auc_full_last_transaction": round(auc_full, 4),
        "auc_full_without_basket_block": round(auc_full_nobasket, 4),
        "foresight_vs_tautology_gap": round(auc_full - auc_early, 4),
        "basket_leakage_contribution_auc": round(auc_full - auc_full_nobasket, 4),
        "lift_decile1_full": round(lift1, 3),
        "basket_block": BASKET,
    }
    (RES / "metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))

    INK, C1, C2, GREY, HI = "#0a1e2e", "#1f77b4", "#8eb600", "#A6B5C2", "#d63031"
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    ax[0].bar(["early\n(first click)", "full\n(end state)", "full minus\nbasket block"],
              [auc_early, auc_full, auc_full_nobasket], color=[C2, C1, GREY])
    for i, v in enumerate([auc_early, auc_full, auc_full_nobasket]):
        ax[0].text(i, v + 0.004, f"{v:.3f}", ha="center", fontweight="bold", color=INK)
    ax[0].axhline(0.5, ls="--", color=HI); ax[0].set_ylim(0.5, max(auc_full, 0.9) * 1.05)
    ax[0].set_ylabel("ROC-AUC (session group-split)")
    ax[0].set_title("Early foresight vs end-of-session tautology", fontweight="bold", color=INK)
    ax[1].plot([0, 1], [0, 1], "--", color="#999"); ax[1].plot(mp, fp, "o-", color=C1)
    ax[1].set_title("Calibration (full model)", fontweight="bold", color=INK)
    ax[1].set_xlabel("predicted P(order)"); ax[1].set_ylabel("observed order rate")
    ax[2].axhline(base * 100, ls="--", color=HI, label=f"population rate {base:.0%}")
    ax[2].bar(["decile 1\n(top scores)"], [yte[order[: len(order) // 10]].mean() * 100], color=C1)
    ax[2].text(0, yte[order[: len(order)//10]].mean()*100 + 1, f"lift x{lift1:.2f}", ha="center", fontweight="bold", color=INK)
    ax[2].set_title("Top-decile order rate (full model)", fontweight="bold", color=INK)
    ax[2].set_ylabel("order rate (%)"); ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.suptitle("E-commerce conversion (DMC 2013, generic webshop): real foresight is modest, "
                 "the end-state lift is largely basket tautology", fontsize=12.5, fontweight="bold",
                 color=INK, x=0.02, ha="left")
    fig.text(0.01, 0.005, f"Source: DMC 2013 prediction-of-orders (generic online shop); {n_sessions:,} sessions, "
             f"session group-split; population order rate {base:.0%} is the curated competition set, not a real funnel · myBytes",
             fontsize=7.5, color="#888")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(FIG / "01_early_vs_late.png", dpi=150, bbox_inches="tight")
    print("saved", FIG / "01_early_vs_late.png")


if __name__ == "__main__":
    main()
