"""Central configuration: paths, feature groups, leakage registry and clinical constants."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "vancomycin-precision-dosing-cohort-400-patients"
SPLITS_FILE = ROOT / "data" / "patient_splits.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"
MODEL_FILE = MODELS_DIR / "vanco_model.joblib"
MODEL_CARD = MODELS_DIR / "model_card.json"

PATIENTS_FILE = RAW_DIR / "vanco_sample_patients.parquet"
REGIMENS_FILE = RAW_DIR / "vanco_sample_dosing_regimens.parquet"
PROFILES_FILE = RAW_DIR / "vanco_sample_pk_profiles.parquet"

SEED = 42

# Outlier exclusion (requested after visual review of SCr vs AUC24): patients with serum
# creatinine > 3.5 mg/dL (6 patients, AUC24 2,400-13,000 mg·h/L, all in the development set)
# are removed from every table. The model is therefore scoped to SCr <= 3.5 mg/dL.
MAX_SCR_MG_DL = 3.5
N_FOLDS = 5
HORIZON_H = 72.0

# ---------------------------------------------------------------------------
# Feature groups (what a clinician actually has at the bedside)
# ---------------------------------------------------------------------------
PATIENT_INPUTS = ["age", "sex", "weight_kg", "height_cm"]
CLINICAL_INPUTS = ["SCr_mg_dl", "CrCL_ml_min", "ICU_flag", "RRT_flag", "ARC_flag"]
DOSING_INPUTS = ["dose_mg", "interval_h", "infusion_duration_h", "loading_dose_mg"]
# Time-dependent PK information: measured (noisy) serum levels with their sampling time.
TDM_INPUTS = ["level_time_h", "level_mg_L"]
PREDICTION_TARGETS = {
    "patient": "true_CL_L_h",  # learned target; AUC24 = daily dose / CL (exact in this simulator)
    "regimen": ["AUC24_true_mg_h_L", "trough_ss_mg_L", "peak_ss_mg_L"],
    "profile": "conc_true_mg_L",
}

# Columns that must never be used as model inputs.
LEAKAGE_REGISTRY = {
    "seed": "Simulator random seed; identifier with no clinical meaning.",
    "true_CL_L_h": "Latent individual clearance; AUC24 = daily dose / true_CL exactly.",
    "true_CL_residual_L_h": "Component of latent clearance.",
    "true_CL_dialysis_L_h": "Component of latent clearance.",
    "true_Vd1_L": "Latent central volume (simulation parameter).",
    "true_Vd2_L": "Latent peripheral volume (simulation parameter).",
    "true_Q_L_h": "Latent inter-compartmental clearance.",
    "true_ke_per_h": "Derived from latent CL/V1.",
    "true_alpha_per_h": "Derived from latent parameters.",
    "true_beta_per_h": "Derived from latent parameters.",
    "true_tHalf_beta_h": "Derived from latent parameters.",
    "true_AUC24_ss_mg_h_L": "Equal to the standard-regimen target.",
    "AUC24_true_mg_h_L": "Prediction target.",
    "AUC24_trapezoidal_mg_h_L": "Numerical integral of the target.",
    "AUC_MIC_ratio": "Target divided by MIC.",
    "trough_ss_mg_L": "Steady-state target derived from latent parameters.",
    "peak_ss_mg_L": "Steady-state target derived from latent parameters.",
    "ss_reached": "Computed from the latent concentration profile.",
    "conc_true_mg_L": "Noise-free latent concentration.",
    "auc_targeted regimen dose": "The auc_targeted regimen was designed with the latent clearance "
    "(log-dose vs log-CL correlation 0.88); its dose must not be used to infer clearance.",
}
REDUNDANT_COLUMNS = {
    "subpopulation": "Deterministic label built from age/BMI/ICU flags.",
    "elderly_flag": "Equals age >= 65.",
    "obesity_flag": "Derived from BMI; BMI is computed from weight and height.",
    "BMI": "Computed from weight and height.",
    "dosing_weight_kg": "Derived from weight and height (adjusted body weight).",
    "dose_mg_kg": "dose_mg / weight_kg.",
    "loading_dose_mg_kg": "loading_dose_mg / weight_kg.",
    "daily_dose_mg": "dose_mg * 24 / interval_h.",
    "n_doses_72h": "Derived from interval_h.",
    "MIC_mg_l": "Affects the AUC/MIC target, not drug exposure.",
}

# ---------------------------------------------------------------------------
# Clinical interpretation (Rybak et al. 2020 consensus; AUC24 assuming MIC 1 mg/L)
# ---------------------------------------------------------------------------
# Accuracy goal for AUC24 (mg·h/L) on held-out patients (revised from 70/70 on 2026-10-06).
GOAL_MAE = 90.0
GOAL_RMSE = 100.0

AUC_TARGET_LOW = 400.0
AUC_TARGET_HIGH = 600.0
AUC_TOXICITY = 800.0  # higher nephrotoxicity risk above this
TARGET_AUC = 500.0
DOSE_ROUNDING_MG = 250.0
ALLOWED_INTERVALS_H = [8.0, 12.0, 24.0, 48.0]
MAX_DAILY_DOSE_MG = 6000.0

# Residual-error model of the assay (combined additive + proportional); refitted in training.
DEFAULT_ERROR_MODEL = (1.5, 0.15)

FEATURE_LABELS = {
    "age": "Age",
    "male": "Sex (male)",
    "log_weight": "Body weight",
    "log_height": "Height",
    "log_bmi": "Body-mass index",
    "log_scr": "Serum creatinine",
    "log_crcl": "Creatinine clearance",
    "icu": "ICU admission",
    "rrt": "Renal replacement therapy",
    "arc": "Augmented renal clearance",
    "crcl_per_kg": "CrCL per kg",
}
