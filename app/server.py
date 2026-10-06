"""VancoDose web application (FastAPI + single static page).

Run:  python -m uvicorn app.server:app --port 8000      (from the project root)
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vanco import config as C
from vanco.analytics import population_analytics
from vanco.data import load_cohort
from vanco.features import regimen_from_row, sample_levels
from vanco.history import PatientJourney
from vanco.predict import InputError, VancoPredictor

log = logging.getLogger("vanco.app")
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="VancoDose", version="2.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class Level(BaseModel):
    time_h: float
    conc_mg_L: float


class PredictRequest(BaseModel):
    patient: dict
    regimen: dict
    levels: list[Level] = Field(default_factory=list)
    target: tuple[float, float] = (C.AUC_TARGET_LOW, C.AUC_TARGET_HIGH)


@lru_cache(maxsize=1)
def predictor() -> VancoPredictor:
    return VancoPredictor.load()


@lru_cache(maxsize=1)
def journeys() -> PatientJourney:
    return PatientJourney(predictor())


class NewCourse(BaseModel):
    start_h: float
    regimen: dict
    condition: dict = Field(default_factory=dict)
    note: str | None = None
    accepted: bool = False
    simulate_levels: bool = False


class NewLevel(BaseModel):
    time_h: float
    conc_mg_L: float


def _journey_call(fn, *a):
    try:
        return fn(*a)
    except KeyError:
        raise HTTPException(404, "No patient with that ID") from None
    except InputError as e:
        raise HTTPException(422, str(e)) from None


@app.get("/api/patients")
def patients():
    return journeys().directory()


@app.get("/api/patients/{pid}")
def patient(pid: str):
    return _journey_call(journeys().journey, pid)


@app.post("/api/patients/{pid}/courses")
def add_course(pid: str, body: NewCourse):
    _journey_call(journeys().add_course, pid, body.model_dump())
    return _journey_call(journeys().journey, pid)


@app.delete("/api/patients/{pid}/courses/{cid}")
def delete_course(pid: str, cid: int):
    _journey_call(journeys().delete_course, pid, cid)
    return _journey_call(journeys().journey, pid)


@app.post("/api/patients/{pid}/levels")
def add_level(pid: str, body: NewLevel):
    _journey_call(journeys().add_level, pid, body.time_h, body.conc_mg_L)
    return _journey_call(journeys().journey, pid)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"app": "VancoDose", "ready": C.MODEL_FILE.exists()}


@app.post("/api/predict")
def predict(req: PredictRequest):
    lo, hi = req.target
    if not 100 <= lo < hi <= 1500:
        raise HTTPException(422, "Target range must satisfy 100 ≤ low < high ≤ 1500 mg·h/L")
    try:
        return predictor().predict(req.patient, req.regimen, [lv.model_dump() for lv in req.levels], (lo, hi))
    except InputError as e:
        raise HTTPException(422, str(e)) from None


@app.get("/api/analytics")
@lru_cache(maxsize=1)
def analytics():
    return population_analytics()


@app.get("/api/model")
@lru_cache(maxsize=1)
def model_info():
    rep = C.REPORTS_DIR
    def table(name):
        f = rep / name
        return json.loads(pd.read_csv(f).round(3).to_json(orient="records")) if f.exists() else []
    return {"card": json.loads(C.MODEL_CARD.read_text()) if C.MODEL_CARD.exists() else {},
            "goal": {"MAE": C.GOAL_MAE, "RMSE": C.GOAL_RMSE},
            "apriori": table("apriori_comparison.csv"), "tdm": table("tdm_comparison.csv"),
            "curves": table("curve_comparison.csv"), "bands": table("test_error_by_auc_band.csv")}


@app.get("/api/examples")
@lru_cache(maxsize=1)
def examples():
    """Held-out test patients (standard regimen + protocol peak/trough) for demonstration."""
    cohort = load_cohort()
    R = cohort.regimen_table
    t = R[(R.split == "test") & (R.regimen_type == "standard")]
    picks = []
    for label, q in [("Reduced kidney function", t.CrCL_ml_min < 45), ("Typical adult", t.CrCL_ml_min.between(80, 110) & (t.ICU_flag == 0)),
                     ("ICU, augmented clearance", t.ARC_flag == 1), ("Obese adult", t.BMI >= 35)]:
        if q.any():
            r = t[q].iloc[0]
            g = regimen_from_row(r)
            lv = sample_levels(cohort.profile(r.regimen_id), g)
            picks.append({"label": label, "id": r.patient_id,
                          "patient": {"age": int(r.age), "sex": r.sex, "weight_kg": round(float(r.weight_kg), 1), "SCr_mg_dl": round(float(r.SCr_mg_dl), 2),
                                      "ICU_flag": int(r.ICU_flag), "RRT_flag": int(r.RRT_flag), "ARC_flag": int(r.ARC_flag)},
                          "regimen": {"dose_mg": round(float(g.dose_mg)), "interval_h": round(g.interval_h, 2),
                                      "infusion_duration_h": round(g.infusion_duration_h, 2), "loading_dose_mg": 0},
                          "levels": [{"time_h": round(a, 2), "conc_mg_L": round(b, 1)} for a, b in lv]})
    return picks
