"""Per-prediction explanations in clinical language.

AUC24 = daily dose / CL, and the clearance model works on log(CL), so SHAP contributions
are additive in log space and become *multiplicative* effects on exposure:

    predicted AUC = typical AUC for this regimen x  prod_i exp(-phi_i)  x  level adjustment

Each factor is reported as "raises / lowers predicted exposure by X%" and the factors
multiply back exactly to the prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_LABELS


class ClearanceExplainer:
    def __init__(self, model, background: pd.DataFrame):
        est = model.steps[-1][1] if hasattr(model, "steps") else model
        if hasattr(est, "coef_"):  # linear pipeline: exact SHAP on the standardised inputs (no shap library needed)
            self._kind, self._model = "linear", model
            self._mean = background.mean()
        else:
            import shap  # only tree/neural models need the shap library (requirements-dev.txt)
            bg = background.sample(min(100, len(background)), random_state=0)
            try:
                self._explainer = shap.TreeExplainer(est, bg)
                self._kind = "tree"
            except Exception:  # noqa: BLE001 - fall back to model-agnostic SHAP
                self._explainer = shap.Explainer(model.predict, bg)
                self._kind = "agnostic"
            self._model = model
        self.base_value = float(np.mean(model.predict(background)))

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        if self._kind == "linear":
            scaler, lin = self._model.steps[0][1], self._model.steps[-1][1]
            return ((X - self._mean) / scaler.scale_).to_numpy() * lin.coef_
        return np.asarray(self._explainer(X).values)

    def explain(self, X: pd.DataFrame, raw: dict, daily_dose: float, map_shift: float = 0.0, n_levels: int = 0) -> dict:
        phi = self.shap_values(X)[0]
        typical_auc = daily_dose / np.exp(self.base_value)
        factors = []
        for name, p in zip(X.columns, phi):
            effect = float(np.exp(-p) - 1)  # relative change in AUC
            factors.append({"feature": name, "label": FEATURE_LABELS.get(name, name), "value": _display_value(name, raw),
                            "effect_pct": effect * 100, "direction": "higher" if effect > 0 else "lower",
                            "log_contribution": float(-p)})
        factors.sort(key=lambda f: abs(f["log_contribution"]), reverse=True)
        out = {"typical_auc": float(typical_auc), "factors": factors, "method": "SHAP (" + self._kind + ")"}
        if n_levels:
            eff = float(np.exp(-map_shift) - 1)
            out["levels_adjustment"] = {"label": f"Measured levels ({n_levels})", "effect_pct": eff * 100,
                                        "direction": "higher" if eff > 0 else "lower",
                                        "text": f"Your measured levels {'raised' if eff > 0 else 'lowered'} the estimate by {abs(eff) * 100:.0f}% compared with the covariate-only prediction."}
        return out


def _display_value(feature: str, raw: dict) -> str:
    m = {"age": f"{raw['age']:.0f} y", "male": "Male" if raw["sex"] == "M" else "Female",
         "log_weight": f"{raw['weight_kg']:.0f} kg", "log_height": f"{raw['height_cm']:.0f} cm",
         "log_bmi": f"{raw['weight_kg'] / (raw['height_cm'] / 100) ** 2:.1f} kg/m²",
         "log_scr": f"{raw['SCr_mg_dl']:.2f} mg/dL", "log_crcl": f"{raw['CrCL_ml_min']:.0f} mL/min",
         "crcl_per_kg": f"{raw['CrCL_ml_min'] / raw['weight_kg']:.2f} mL/min/kg",
         "icu": "Yes" if raw["ICU_flag"] else "No", "rrt": "Yes" if raw["RRT_flag"] else "No", "arc": "Yes" if raw["ARC_flag"] else "No"}
    return m.get(feature, "")
