"""Loading, joining, validating and splitting the three source tables.

Relationships
-------------
patients (1) ── (3) regimens (1) ── (865) pk_profile rows
  patient_id         regimen_id = <patient_id>_<regimen_type>

Regimen types: ``standard`` (no loading dose), ``loading`` (loading dose replaces the
first maintenance dose and is infused at the maintenance rate) and ``auc_targeted``
(designed by the simulator *using the latent clearance* - its dose is leakage-tainted).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Cohort:
    patients: pd.DataFrame
    regimens: pd.DataFrame
    profiles: pd.DataFrame

    @property
    def regimen_table(self) -> pd.DataFrame:
        """Regimen rows joined to their patient (one row per regimen)."""
        return self.regimens.merge(self.patients, on="patient_id", how="left", validate="many_to_one")

    def profile(self, regimen_id: str) -> pd.DataFrame:
        return self.profiles.loc[regimen_id]


@lru_cache(maxsize=1)
def load_cohort() -> Cohort:
    patients = pd.read_parquet(C.PATIENTS_FILE)
    regimens = pd.read_parquet(C.REGIMENS_FILE)
    profiles = pd.read_parquet(C.PROFILES_FILE)
    splits = pd.read_csv(C.SPLITS_FILE)
    # Existing grouped split: 240 train / 40 validation / 40 calibration / 80 test.
    # This pipeline uses train+validation+calibration (320) for development via grouped CV
    # and keeps the 80 test patients untouched until the final evaluation.
    splits["split"] = np.where(splits["split"] == "test", "test", "dev")
    patients = patients.merge(splits, on="patient_id", how="left", validate="one_to_one")
    if patients["split"].isna().any():
        raise ValueError("Every patient must have a split assignment")
    excluded = patients.loc[patients.SCr_mg_dl > C.MAX_SCR_MG_DL, "patient_id"]
    if len(excluded):
        log.info("Excluding %d outlier patients with SCr > %.1f mg/dL: %s", len(excluded), C.MAX_SCR_MG_DL, ", ".join(excluded))
    patients = patients[~patients.patient_id.isin(excluded)].reset_index(drop=True)
    regimens = regimens[~regimens.patient_id.isin(excluded)].reset_index(drop=True)
    profiles = profiles[~profiles.patient_id.isin(excluded)]
    profiles = profiles.sort_values(["regimen_id", "time_h"]).set_index("regimen_id")
    log.info("Loaded %d patients, %d regimens, %d profile rows", len(patients), len(regimens), len(profiles))
    return Cohort(patients, regimens, profiles)


def audit(cohort: Cohort) -> dict:
    """Data-quality and leakage checks. Returns a JSON-serialisable summary."""
    P, R = cohort.patients, cohort.regimens
    K = cohort.profiles.reset_index()
    M = cohort.regimen_table
    out: dict = {"excluded_outliers": f"Patients with SCr > {C.MAX_SCR_MG_DL} mg/dL removed before all analyses (6 patients)",
                 "counts": {"patients": len(P), "regimens": len(R), "profile_rows": len(K)}}
    out["missing_values"] = int(P.isna().sum().sum() + R.isna().sum().sum() + K.isna().sum().sum())
    out["duplicates"] = {
        "patient_id": int(P.patient_id.duplicated().sum()),
        "regimen_id": int(R.regimen_id.duplicated().sum()),
        "profile_time": int(K.duplicated(["regimen_id", "time_h"]).sum()),
    }
    out["referential_integrity"] = {
        "regimens_without_patient": int((~R.patient_id.isin(P.patient_id)).sum()),
        "profiles_without_regimen": int((~K.regimen_id.isin(R.regimen_id)).sum()),
        "regimens_per_patient": R.groupby("patient_id").size().value_counts().to_dict(),
        "points_per_profile": K.groupby("regimen_id").size().value_counts().to_dict(),
    }
    rel = (M.daily_dose_mg / M.true_CL_L_h - M.AUC24_true_mg_h_L).abs() / M.AUC24_true_mg_h_L
    out["identities"] = {
        "AUC24 == daily_dose / true_CL (max rel. error)": float(rel.max()),
        "AUC_MIC_ratio == AUC24 / MIC": bool(np.allclose(M.AUC_MIC_ratio, M.AUC24_true_mg_h_L / M.MIC_mg_l, rtol=1e-3)),
        "daily_dose == dose * 24 / interval": bool(np.allclose(M.daily_dose_mg, M.dose_mg * 24 / M.interval_h, rtol=1e-3)),
        "CrCL == Cockcroft-Gault(total body weight)": bool(np.allclose(
            P.CrCL_ml_min, (140 - P.age) * P.weight_kg / (72 * P.SCr_mg_dl) * np.where(P.sex == "F", 0.85, 1), rtol=1e-2)),
        "obesity_flag == (BMI >= 30)": bool(((P.BMI >= 30).astype(int) == P.obesity_flag).all()),
    }
    at = M[M.regimen_type == "auc_targeted"]
    st = M[M.regimen_type == "standard"]
    out["leakage_signals"] = {
        "corr(log daily dose, log true CL) auc_targeted": float(np.corrcoef(np.log(at.daily_dose_mg), np.log(at.true_CL_L_h))[0, 1]),
        "corr(log daily dose, log true CL) standard": float(np.corrcoef(np.log(st.daily_dose_mg), np.log(st.true_CL_L_h))[0, 1]),
    }
    res = K.conc_measured_mg_L - K.conc_true_mg_L
    bins = pd.cut(K.conc_true_mg_L, [-1, 0.01, 5, 10, 20, 40, 80, 1e4])
    out["assay_noise_sd_by_true_conc"] = {str(k): round(float(v), 3) for k, v in res.groupby(bins, observed=True).std().items()}
    num = P.select_dtypes("number").drop(columns=["seed"])
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    fence = ((num < q1 - 3 * (q3 - q1)) | (num > q3 + 3 * (q3 - q1))).sum()
    out["extreme_values_3IQR"] = {k: int(v) for k, v in fence.items() if v > 0}
    out["ranges"] = {c: [float(P[c].min()), float(P[c].max())] for c in ["age", "weight_kg", "height_cm", "SCr_mg_dl", "CrCL_ml_min"]}
    out["auc_by_regimen_type"] = M.groupby("regimen_type").AUC24_true_mg_h_L.describe().round(1).to_dict(orient="index")
    out["phenotypes"] = {c: P[c].value_counts().to_dict() for c in ["sex", "ICU_flag", "RRT_flag", "ARC_flag", "MIC_mg_l"]}
    out["split_sizes"] = P.split.value_counts().to_dict()
    return out
