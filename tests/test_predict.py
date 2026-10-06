import numpy as np
import pytest

from vanco import config as C
from vanco.predict import InputError, VancoPredictor

pytestmark = pytest.mark.skipif(not C.MODEL_FILE.exists(), reason="train the model first: python -m vanco.train")
PATIENT = {"age": 60, "sex": "M", "weight_kg": 80, "SCr_mg_dl": 1.0, "ICU_flag": 0, "RRT_flag": 0, "ARC_flag": 0}
REGIMEN = {"dose_mg": 1000, "interval_h": 12, "infusion_duration_h": 1}


@pytest.fixture(scope="module")
def model():
    return VancoPredictor.load()


def test_auc_is_dose_proportional(model):
    a = model.predict(PATIENT, REGIMEN)["auc24"]
    b = model.predict(PATIENT, {**REGIMEN, "dose_mg": 2000})["auc24"]
    assert b == pytest.approx(2 * a, rel=1e-6)


def test_worse_kidney_function_means_higher_exposure(model):
    good = model.predict(PATIENT, REGIMEN)["auc24"]
    poor = model.predict({**PATIENT, "SCr_mg_dl": 3.0}, REGIMEN)["auc24"]
    assert poor > good * 1.5


def test_higher_levels_raise_estimate(model):
    low = model.predict(PATIENT, REGIMEN, [{"time_h": 25, "conc_mg_L": 20}, {"time_h": 36, "conc_mg_L": 6}])["auc24"]
    high = model.predict(PATIENT, REGIMEN, [{"time_h": 25, "conc_mg_L": 40}, {"time_h": 36, "conc_mg_L": 20}])["auc24"]
    assert high > low


def test_explanation_multiplies_back(model):
    r = model.predict(PATIENT, REGIMEN)
    ex = r["explanation"]
    total = ex["typical_auc"] * np.exp(sum(f["log_contribution"] for f in ex["factors"]))
    assert total == pytest.approx(ex["prior_auc24"], rel=1e-6)
    assert r["auc24"] == pytest.approx(ex["prior_auc24"], rel=1e-9)  # no levels -> prediction equals prior


def test_recommendation_hits_target(model):
    rec = model.predict(PATIENT, REGIMEN)["recommendation"]["best"]
    assert 400 <= rec["auc24"] <= 600 and rec["dose_mg"] % 250 == 0


@pytest.mark.parametrize("bad", [{"age": 5}, {"sex": "X"}, {"weight_kg": "abc"}])
def test_invalid_input(model, bad):
    with pytest.raises(InputError):
        model.predict({**PATIENT, **bad}, REGIMEN)


def test_api():
    from fastapi.testclient import TestClient
    from app.server import app
    c = TestClient(app)
    assert c.post("/api/predict", json={"patient": PATIENT, "regimen": REGIMEN}).status_code == 200
    assert c.post("/api/predict", json={"patient": {**PATIENT, "age": 3}, "regimen": REGIMEN}).status_code == 422
    assert c.get("/api/analytics").status_code == 200


def test_patient_journey_add_and_undo(tmp_path):
    from vanco.history import HistoryStore, PatientJourney
    j = PatientJourney(VancoPredictor.load(), HistoryStore(tmp_path / "h.db"))
    pid = j.directory()[0]["id"]
    first = j.journey(pid)
    assert len(first["courses"]) == 1 and first["courses"][0]["response"]  # dataset course with peak/trough
    best = first["next"]["recommendation"]["best"]
    cid = j.add_course(pid, {"start_h": first["next"]["start_h"], "accepted": True, "simulate_levels": True,
                             "regimen": {"dose_mg": best["dose_mg"], "interval_h": best["interval_h"]}})
    second = j.journey(pid)
    c2 = second["courses"][1]
    assert c2["source"] == "recommended" and c2["response"] and len(second["levels"]) == 4
    assert c2["prediction"]["basis_levels"] == 2 and any("Course 1" in s for s in c2["reason"])
    with pytest.raises(InputError):  # timeline must move forward
        j.add_course(pid, {"start_h": 1, "regimen": {"dose_mg": 500, "interval_h": 12}})
    j.delete_course(pid, cid)
    assert len(j.journey(pid)["courses"]) == 1 and len(j.journey(pid)["levels"]) == 2


def test_patient_api_404():
    from fastapi.testclient import TestClient
    from app.server import app
    assert TestClient(app).get("/api/patients/NOPE").status_code == 404
