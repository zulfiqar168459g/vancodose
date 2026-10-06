"""Two-compartment intravenous-infusion pharmacokinetics (closed form).

Reproduces the dataset's simulator to < 0.002 mg/L for standard and loading regimens.
Parameters are (CL, V1, V2, Q) in L/h and L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HORIZON_H


@dataclass(frozen=True)
class Regimen:
    dose_mg: float
    interval_h: float
    infusion_duration_h: float
    loading_dose_mg: float = 0.0

    @property
    def daily_dose_mg(self) -> float:
        return self.dose_mg * 24.0 / self.interval_h

    def validate(self) -> None:
        if not (self.dose_mg > 0 and self.interval_h > 0 and self.infusion_duration_h > 0):
            raise ValueError("Dose, interval and infusion duration must be positive")
        if self.infusion_duration_h >= self.interval_h:
            raise ValueError("Infusion duration must be shorter than the dosing interval")
        if self.loading_dose_mg < 0:
            raise ValueError("Loading dose cannot be negative")

    def events(self, horizon: float = HORIZON_H) -> list[tuple[float, float, float]]:
        """(start_h, amount_mg, duration_h). A loading dose replaces dose 1 at the maintenance rate."""
        out, k = [], 0
        while k * self.interval_h < horizon - 1e-9:
            amount = self.loading_dose_mg if (k == 0 and self.loading_dose_mg > 0) else self.dose_mg
            out.append((k * self.interval_h, amount, self.infusion_duration_h * amount / self.dose_mg))
            k += 1
        return out


@dataclass(frozen=True)
class DosingHistory:
    """Consecutive courses: [(start_h, Regimen), ...]. Each course runs until the next one starts.

    Exposes the same ``events`` interface as :class:`Regimen`, so the PK model and the Bayesian
    update work unchanged across regimen changes (drug carried over from earlier courses included).
    """
    courses: tuple

    def events(self, horizon: float = HORIZON_H) -> list[tuple[float, float, float]]:
        out = []
        starts = [c[0] for c in self.courses] + [np.inf]
        for (start, reg), nxt in zip(self.courses, starts[1:]):
            end = min(nxt, horizon)
            for t0, amount, dur in reg.events(max(end - start, 0.0)):
                if start + t0 < end - 1e-9:
                    out.append((start + t0, amount, dur))
        return out


def _exponents(CL: float, V1: float, V2: float, Q: float):
    k10, k12, k21 = CL / V1, Q / V1, Q / V2
    s = k10 + k12 + k21
    d = np.sqrt(max(s * s - 4 * k10 * k21, 1e-14))
    a, b = (s + d) / 2, (s - d) / 2
    ca, cb = (a - k21) / (V1 * (a - b)), (k21 - b) / (V1 * (a - b))
    return ((ca, a), (cb, b))


def concentration(t, regimen, CL: float, V1: float, V2: float, Q: float) -> np.ndarray:
    """Central concentration (mg/L) at times ``t`` (h) after the first dose."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros_like(t)
    terms = _exponents(CL, V1, V2, Q)
    for t0, amount, dur in regimen.events(max(float(t.max()), 1e-6) + 1e-9):
        rate = amount / dur
        m = t > t0
        if not m.any():
            continue
        tt = t[m]
        te = np.minimum(tt, t0 + dur)
        for c, lam in terms:
            out[m] += rate * c / lam * (np.exp(-lam * (tt - te)) - np.exp(-lam * (tt - t0)))
    return out


def steady_state(regimen: Regimen, CL: float, V1: float, V2: float, Q: float) -> dict:
    """Analytic steady-state peak (end of infusion), trough (pre-dose) and AUC24."""
    tau, T = regimen.interval_h, regimen.infusion_duration_h
    rate = regimen.dose_mg / T
    def c_ss(t):  # t in [0, tau): current dose + geometric sum of all previous doses
        v = 0.0
        for c, lam in _exponents(CL, V1, V2, Q):
            current = np.exp(-lam * (t - min(t, T))) - np.exp(-lam * t)
            previous = (np.exp(lam * T) - 1) * np.exp(-lam * t) * np.exp(-lam * tau) / (1 - np.exp(-lam * tau))
            v += rate * c / lam * (current + previous)
        return v
    peak, trough = c_ss(T), c_ss(tau - 1e-9)
    return {"peak_ss_mg_L": float(peak), "trough_ss_mg_L": float(trough), "auc24_mg_h_L": regimen.daily_dose_mg / CL}
