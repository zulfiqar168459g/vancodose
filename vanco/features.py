"""Clinically meaningful feature engineering and TDM sampling designs.

Only information available at the bedside is used: demographics, renal function,
ICU/RRT/ARC status, the regimen, and (optionally) measured serum levels with times.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pk import Regimen

# Candidate engineered features; the ablation in experiments.py showed that the final six
# carry all of the predictive information (height, BMI, sex, SCr and CrCL/kg add nothing once
# Cockcroft-Gault CrCL and weight are known).
CANDIDATE_FEATURES = ["age", "male", "log_weight", "log_height", "log_bmi", "log_scr", "log_crcl",
                      "crcl_per_kg", "icu", "rrt", "arc"]
PATIENT_FEATURES = ["age", "log_weight", "log_crcl", "icu", "rrt", "arc"]


def patient_features(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Patient-level model matrix from raw bedside inputs (one row per patient)."""
    cols = PATIENT_FEATURES if columns is None else columns
    w, crcl = df["weight_kg"].astype(float), df["CrCL_ml_min"].astype(float)
    h = df["height_cm"].astype(float) if "height_cm" in df else pd.Series(np.nan, index=df.index)
    return pd.DataFrame({
        "age": df["age"].astype(float),
        "male": (df["sex"] == "M").astype(float) if "sex" in df else np.nan,
        "log_weight": np.log(w),
        "log_height": np.log(h),
        "log_bmi": np.log(w / (h / 100) ** 2),
        "log_scr": np.log(df["SCr_mg_dl"].astype(float)),
        "log_crcl": np.log(crcl),
        "crcl_per_kg": crcl / w,
        "icu": df["ICU_flag"].astype(float),
        "rrt": df["RRT_flag"].astype(float),
        "arc": df["ARC_flag"].astype(float),
    }, index=df.index)[cols]


def regimen_from_row(row) -> Regimen:
    return Regimen(float(row["dose_mg"]), float(row["interval_h"]), float(row["infusion_duration_h"]),
                   float(row.get("loading_dose_mg", 0.0) or 0.0))


# ---------------------------------------------------------------------------
# TDM sampling: what a hospital would actually draw
# ---------------------------------------------------------------------------
def _nearest_before(t: np.ndarray, target: float) -> int:
    return int(np.clip(np.searchsorted(t, target, side="right") - 1, 0, len(t) - 1))


def sample_levels(profile: pd.DataFrame, regimen: Regimen, design: str = "peak_trough",
                  rng: np.random.Generator | None = None) -> list[tuple[float, float]]:
    """Return measured (time_h, mg/L) levels from a 5-minute noisy profile.

    ``peak_trough``: peak 1 h after the end of infusion of dose k and trough just before
    dose k+1 (protocol: k = 3, i.e. around 24-36 h). ``trough``: the trough only.
    With ``rng`` the dose number and sampling times are jittered (training augmentation:
    different grid points carry independent assay noise).
    """
    t = profile["time_h"].to_numpy()
    y = profile["conc_measured_mg_L"].to_numpy()
    tau, T = regimen.interval_h, regimen.infusion_duration_h
    n_doses = int(np.ceil(72.0 / tau))
    if rng is None:
        k, dt_trough, dt_peak = 3, 1e-6, 1.0
    else:
        k = int(rng.integers(2, n_doses))
        dt_trough, dt_peak = rng.uniform(0.0, 0.5), rng.uniform(0.5, 2.0)
    trough_t = k * tau - dt_trough
    peak_t = (k - 1) * tau + T + dt_peak
    if design == "two_pairs":  # a second peak/trough pair three doses later (or the last full interval)
        k2 = min(k + 3, n_doses - 1)
        idx = [_nearest_before(t, peak_t), _nearest_before(t, trough_t),
               _nearest_before(t, (k2 - 1) * tau + T + dt_peak), _nearest_before(t, k2 * tau - dt_trough)]
        idx = sorted(set(idx))
    elif design == "trough":
        idx = [_nearest_before(t, trough_t)]
    else:
        idx = [_nearest_before(t, peak_t), _nearest_before(t, trough_t)]
    return [(float(t[i]), float(y[i])) for i in idx]


def level_features(levels: list[tuple[float, float]], regimen: Regimen) -> dict:
    """Classical pharmacist features from a peak/trough pair (Sawchuk-Zaske style)."""
    (tp, cp), (tt, ct) = levels[0], levels[-1]
    cp, ct = max(cp, 0.5), max(ct, 0.5)
    ke = np.log(cp / ct) / max(tt - tp, 0.5) if len(levels) > 1 else np.nan
    return {
        "log_peak": np.log(cp) if len(levels) > 1 else np.nan,
        "log_trough": np.log(ct),
        "log_trough_per_dose": np.log(ct / regimen.dose_mg),
        "ke_sz": ke,
        "t_trough": tt,
        "log_dose": np.log(regimen.dose_mg),
        "interval": regimen.interval_h,
        "infusion": regimen.infusion_duration_h,
        "loading": float(regimen.loading_dose_mg > 0),
    }
