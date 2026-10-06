"""Production inference: patient + regimen (+ optional levels) -> exposure, interpretation,
recommended regimen and explanation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import erf, sqrt

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .evaluate import interpret
from .features import patient_features
from .models import bayesian_update
from .pk import Regimen, _exponents, concentration, steady_state

log = logging.getLogger(__name__)
Z80 = 1.2816


class InputError(ValueError):
    """Raised for invalid user input (shown to the user verbatim)."""


def cockcroft_gault(age: float, weight_kg: float, scr: float, sex: str) -> float:
    return (140 - age) * weight_kg / (72 * scr) * (0.85 if sex == "F" else 1.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


@dataclass
class Bundle:
    population: object
    explainer: object
    input_ranges: dict
    card: dict


class VancoPredictor:
    def __init__(self, bundle: Bundle):
        self.b = bundle
        self.pop = bundle.population

    @classmethod
    def load(cls, path=C.MODEL_FILE) -> "VancoPredictor":
        return cls(joblib.load(path))

    # ------------------------------------------------------------------ inputs
    def _patient(self, p: dict) -> dict:
        try:
            raw = {"age": float(p["age"]), "sex": str(p["sex"]).upper()[:1], "weight_kg": float(p["weight_kg"]),
                   "height_cm": float(p.get("height_cm") or 170.0), "SCr_mg_dl": float(p["SCr_mg_dl"]),
                   "ICU_flag": int(bool(p.get("ICU_flag", 0))), "RRT_flag": int(bool(p.get("RRT_flag", 0))),
                   "ARC_flag": int(bool(p.get("ARC_flag", 0)))}
        except (KeyError, TypeError, ValueError) as e:
            raise InputError(f"Missing or invalid patient field: {e}") from None
        if raw["sex"] not in ("M", "F"):
            raise InputError("Sex must be M or F")
        for k, lo, hi in [("age", 18, 110), ("weight_kg", 25, 300), ("SCr_mg_dl", 0.1, 15)]:
            if not lo <= raw[k] <= hi:
                raise InputError(f"{k} must be between {lo} and {hi}")
        crcl = p.get("CrCL_ml_min")
        raw["CrCL_ml_min"] = float(crcl) if crcl not in (None, "") else cockcroft_gault(raw["age"], raw["weight_kg"], raw["SCr_mg_dl"], raw["sex"])
        return raw

    @staticmethod
    def _regimen(r: dict) -> Regimen:
        try:
            g = Regimen(float(r["dose_mg"]), float(r["interval_h"]), float(r.get("infusion_duration_h") or max(1.0, float(r["dose_mg"]) / 1000)),
                        float(r.get("loading_dose_mg") or 0.0))
            g.validate()
        except (KeyError, TypeError) as e:
            raise InputError(f"Missing or invalid dosing field: {e}") from None
        except ValueError as e:
            raise InputError(str(e)) from None
        return g

    @staticmethod
    def _levels(levels) -> list[tuple[float, float]]:
        out = []
        for lv in levels or []:
            t, c = float(lv["time_h"]), float(lv["conc_mg_L"])
            if not (0 < t <= 72 and 0 <= c < 200):
                raise InputError("Levels need a time between 0 and 72 h after the first dose and a concentration in mg/L")
            out.append((t, c))
        return sorted(out)

    # -------------------------------------------------------------- prediction
    def predict(self, patient: dict, regimen: dict, levels=None, target=(C.AUC_TARGET_LOW, C.AUC_TARGET_HIGH)) -> dict:
        raw, g, lv = self._patient(patient), self._regimen(regimen), self._levels(levels)
        X = patient_features(pd.DataFrame([raw]))
        prior = self.pop.prior(X)[0]
        est = bayesian_update(prior, self.pop.omega, g, lv, self.pop.error_model)
        CL, V1, V2, Q = est.params
        sd = est.log_cl_sd
        auc = g.daily_dose_mg / CL
        lo, hi = g.daily_dose_mg / np.exp(est.theta[0] + Z80 * sd), g.daily_dose_mg / np.exp(est.theta[0] - Z80 * sd)
        ss = steady_state(g, CL, V1, V2, Q)
        grid = np.round(np.arange(0, 72.0001, 0.25), 2)
        curve = concentration(grid, g, CL, V1, V2, Q)
        p_target = self._p_between(np.log(g.daily_dose_mg) - est.theta[0], sd, *target)
        result = {
            "mode": "bayesian" if lv else "a_priori",
            "auc24": float(auc), "auc24_interval_80": [float(lo), float(hi)],
            "probability_in_target": p_target,
            "peak_ss": ss["peak_ss_mg_L"], "trough_ss": ss["trough_ss_mg_L"],
            "clearance_L_h": float(CL), "half_life_h": float(np.log(2) / _exponents(CL, V1, V2, Q)[1][1]),
            "volumes_L": {"V1": float(V1), "V2": float(V2)},
            "interpretation": interpret(auc), "target": list(target),
            "regimen": {**g.__dict__, "daily_dose_mg": g.daily_dose_mg},
            "curve": [{"t": float(t), "c": float(c)} for t, c in zip(grid, curve)],
            "levels": [{"time_h": t, "conc_mg_L": c, "fitted_mg_L": float(concentration([t], g, CL, V1, V2, Q)[0])} for t, c in lv],
            "patient": raw,
            "warnings": self._warnings(raw),
        }
        result["recommendation"] = self.recommend(est, raw, target, current=g)
        best = result["recommendation"]["best"]
        if best:
            rg = Regimen(best["dose_mg"], best["interval_h"], best["infusion_duration_h"], result["recommendation"]["loading_dose_mg"] or 0.0)
            result["rec_curve"] = [{"t": float(t), "c": float(c)} for t, c in zip(grid, concentration(grid, rg, CL, V1, V2, Q))]
        result["explanation"] = self.b.explainer.explain(X, raw, g.daily_dose_mg, est.theta[0] - prior[0], len(lv))
        result["explanation"]["prior_auc24"] = float(g.daily_dose_mg / np.exp(prior[0]))
        return result

    @staticmethod
    def _p_between(log_auc_mean, sd, low, high) -> float:
        return float(_norm_cdf((np.log(high) - log_auc_mean) / sd) - _norm_cdf((np.log(low) - log_auc_mean) / sd))

    def recommend(self, est, raw, target=(C.AUC_TARGET_LOW, C.AUC_TARGET_HIGH), current: Regimen | None = None) -> dict:
        """Practical regimen (doses rounded to 250 mg; 8/12/24 h) aiming for the target mid-point."""
        CL, V1, V2, Q = est.params
        goal = float(np.sqrt(target[0] * target[1])) if target != (C.AUC_TARGET_LOW, C.AUC_TARGET_HIGH) else C.TARGET_AUC
        options = []
        for tau in C.ALLOWED_INTERVALS_H:
            dose = max(C.DOSE_ROUNDING_MG, round(goal * CL * tau / 24 / C.DOSE_ROUNDING_MG) * C.DOSE_ROUNDING_MG)
            cap = np.floor(C.MAX_DAILY_DOSE_MG * tau / 24 / C.DOSE_ROUNDING_MG) * C.DOSE_ROUNDING_MG
            capped = dose > cap
            dose = float(min(dose, cap))
            g = Regimen(dose, tau, max(1.0, dose / 1000.0))
            ss = steady_state(g, CL, V1, V2, Q)
            auc = g.daily_dose_mg / CL
            p = self._p_between(np.log(g.daily_dose_mg) - est.theta[0], est.log_cl_sd, *target)
            options.append({"dose_mg": dose, "interval_h": tau, "infusion_duration_h": g.infusion_duration_h,
                            "daily_dose_mg": g.daily_dose_mg, "auc24": auc, "peak_ss": ss["peak_ss_mg_L"],
                            "trough_ss": ss["trough_ss_mg_L"], "probability_in_target": p,
                            "label": f"{dose:,.0f} mg every {tau:.0f} h", "capped_at_max_daily_dose": bool(capped)})
        # Prefer the highest probability of target attainment; q12h breaks near-ties (common practice).
        options.sort(key=lambda o: (-round(o["probability_in_target"], 2), abs(o["interval_h"] - 12)))
        best = options[0]
        loading = None
        if not est.n_levels:  # starting therapy: 25 mg/kg actual body weight, max 3 g
            loading = float(min(3000.0, round(25 * raw["weight_kg"] / C.DOSE_ROUNDING_MG) * C.DOSE_ROUNDING_MG))
        change = None
        if current is not None:
            change = (best["daily_dose_mg"] - current.daily_dose_mg) / current.daily_dose_mg * 100
        note = None
        if best["capped_at_max_daily_dose"]:
            note = (f"Even {C.MAX_DAILY_DOSE_MG:,.0f} mg/day is predicted to fall short of the target; consider continuous infusion "
                    "and specialist (infectious diseases / pharmacy) input.")
        elif best["auc24"] > target[1] * 1.15:
            note = "The smallest practical dose still exceeds the target; consider extending the interval and confirm with levels."
        return {"best": best, "options": options, "note": note, "loading_dose_mg": loading, "goal_auc24": goal, "daily_dose_change_pct": change,
                "basis": "Bayesian estimate with measured levels" if est.n_levels else "Covariate-only estimate (no levels yet)"}

    def _warnings(self, raw: dict) -> list[str]:
        w = []
        for k, (lo, hi) in self.b.input_ranges.items():
            if not lo <= raw[k] <= hi:
                w.append(f"{k.split('_')[0]} = {raw[k]:g} lies outside the training data range ({lo:g}–{hi:g}); treat the estimate with caution.")
        if raw["RRT_flag"]:
            w.append("Only 9 patients on renal replacement therapy were in the training data; measured levels are strongly advised.")
        return w
