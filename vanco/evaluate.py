"""Metrics and clinical interpretation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    e = y_pred - y_true
    ss_res, ss_tot = float(np.sum(e ** 2)), float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "R2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "MdAPE_%": float(np.median(np.abs(e) / np.abs(y_true)) * 100),
        "bias": float(np.mean(e)),
        "n": int(len(e)),
    }


def auc_category(auc) -> np.ndarray:
    auc = np.asarray(auc, float)
    return np.where(auc < C.AUC_TARGET_LOW, "below", np.where(auc <= C.AUC_TARGET_HIGH, "within", "above"))


def category_agreement(y_true, y_pred) -> float:
    return float(np.mean(auc_category(y_true) == auc_category(y_pred)))


def by_band(y_true, y_pred, bands=(0, 400, 600, 800, 1500, np.inf)) -> pd.DataFrame:
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    df["band"] = pd.cut(df["true"], list(bands), right=True)
    rows = []
    for band, g in df.groupby("band", observed=True):
        m = regression_metrics(g["true"], g["pred"])
        lo, hi = band.left, band.right
        label = f"≤{hi:.0f}" if lo == 0 else (f">{lo:.0f}" if np.isinf(hi) else f"{lo:.0f}–{hi:.0f}")
        rows.append({"true AUC band": label, "n": m["n"], "MAE": m["MAE"], "RMSE": m["RMSE"], "MdAPE_%": m["MdAPE_%"]})
    return pd.DataFrame(rows)


def interpret(auc: float) -> dict:
    """Plain-language clinical category for a predicted AUC24 (MIC 1 mg/L assumed)."""
    if auc < C.AUC_TARGET_LOW:
        return {"category": "below", "label": "Below target",
                "message": f"Predicted exposure is below the {C.AUC_TARGET_LOW:.0f}-{C.AUC_TARGET_HIGH:.0f} mg·h/L target; efficacy may be insufficient."}
    if auc <= C.AUC_TARGET_HIGH:
        return {"category": "within", "label": "Within target",
                "message": f"Predicted exposure is within the {C.AUC_TARGET_LOW:.0f}-{C.AUC_TARGET_HIGH:.0f} mg·h/L target range."}
    if auc <= C.AUC_TOXICITY:
        return {"category": "above", "label": "Above target",
                "message": "Predicted exposure exceeds the target range; consider a dose reduction."}
    return {"category": "toxic", "label": "Above target, toxicity concern",
            "message": f"Predicted exposure exceeds {C.AUC_TOXICITY:.0f} mg·h/L, associated with a higher risk of acute kidney injury."}
