"""Population analytics for the dashboard (standard weight-based dosing in the cohort)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .data import load_cohort

CATS = ["below", "within", "above", "toxic"]


def _cat(auc):
    return pd.cut(auc, [0, C.AUC_TARGET_LOW, C.AUC_TARGET_HIGH, C.AUC_TOXICITY, np.inf], labels=CATS, right=False)


def _attain(df):
    s = df["category"].value_counts(normalize=True).reindex(CATS, fill_value=0) * 100
    return {**{k: round(float(v), 1) for k, v in s.items()}, "n": int(len(df)), "median_auc": round(float(df.AUC24_true_mg_h_L.median()), 0)}


def population_analytics() -> dict:
    cohort = load_cohort()
    R = cohort.regimen_table
    std = R[R.regimen_type == "standard"].copy()
    std["category"] = _cat(std.AUC24_true_mg_h_L)
    std["renal"] = pd.cut(std.CrCL_ml_min, [0, 30, 60, 90, 130, np.inf], labels=["<30", "30–60", "60–90", "90–130", "≥130"], right=False)
    edges = [0, 200, 300, 400, 500, 600, 700, 800, 1000, 1200, 1500, 2000, 3000, np.inf]
    labels = ["<200", "200–300", "300–400", "400–500", "500–600", "600–700", "700–800", "800–1000", "1000–1200", "1200–1500", "1500–2000", "2000–3000", "≥3000"]
    hist = pd.cut(std.AUC24_true_mg_h_L, edges, labels=labels, right=False).value_counts().reindex(labels, fill_value=0)
    groups = {"All patients": std, "ICU": std[std.ICU_flag == 1], "Non-ICU": std[std.ICU_flag == 0],
              "Obese (BMI ≥30)": std[std.BMI >= 30], "Elderly (≥65 y)": std[std.age >= 65],
              "Augmented renal clearance": std[std.ARC_flag == 1], "Renal replacement therapy": std[std.RRT_flag == 1]}
    risk = []
    for name, g in std.groupby("renal", observed=True):
        risk.append({"group": f"CrCL {name} mL/min", **_attain(g)})
    sample = std.sample(min(400, len(std)), random_state=0)
    out = {
        "n_patients": int(len(std)),
        "regimen_note": "Standard weight-based regimen (15–20 mg/kg, no loading dose) from the synthetic cohort.",
        "auc_histogram": {"labels": labels, "counts": hist.astype(int).tolist()},
        "overall": _attain(std),
        "subgroups": [{"group": k, **_attain(g)} for k, g in groups.items() if len(g)],
        "renal_risk": risk,
        "peak_trough": [{"trough": round(float(r.trough_ss_mg_L), 1), "peak": round(float(r.peak_ss_mg_L), 1),
                         "auc": round(float(r.AUC24_true_mg_h_L), 0), "cat": str(r["category"])} for _, r in sample.iterrows()],
        "trough_auc_correlation": round(float(np.corrcoef(np.log(std.trough_ss_mg_L.clip(lower=0.1)), np.log(std.AUC24_true_mg_h_L))[0, 1]), 3),
    }
    te = C.REPORTS_DIR / "test_evaluation.json"
    if te.exists():
        out["model_test"] = json.loads(te.read_text())
    return out
