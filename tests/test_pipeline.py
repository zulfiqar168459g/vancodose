import numpy as np
import pandas as pd
import pytest

from vanco import config as C
from vanco.data import load_cohort
from vanco.features import PATIENT_FEATURES, patient_features, regimen_from_row, sample_levels
from vanco.pk import Regimen, concentration, steady_state


@pytest.fixture(scope="module")
def cohort():
    return load_cohort()


def test_simulator_reproduces_dataset(cohort):
    R = cohort.regimen_table
    for kind in ("standard", "loading"):
        r = R[R.regimen_type == kind].iloc[3]
        prof = cohort.profile(r.regimen_id)
        c = concentration(prof.time_h, regimen_from_row(r), r.true_CL_L_h, r.true_Vd1_L, r.true_Vd2_L, r.true_Q_L_h)
        assert np.max(np.abs(c - prof.conc_true_mg_L.to_numpy())) < 0.01
        ss = steady_state(regimen_from_row(r), r.true_CL_L_h, r.true_Vd1_L, r.true_Vd2_L, r.true_Q_L_h)
        assert ss["peak_ss_mg_L"] == pytest.approx(r.peak_ss_mg_L, rel=1e-3)
        assert ss["auc24_mg_h_L"] == pytest.approx(r.AUC24_true_mg_h_L, rel=1e-4)


def test_features_use_only_bedside_inputs():
    bedside = pd.DataFrame([{"age": 60, "sex": "F", "weight_kg": 70, "SCr_mg_dl": 1.0, "CrCL_ml_min": 60,
                             "ICU_flag": 0, "RRT_flag": 0, "ARC_flag": 0}])
    X = patient_features(bedside)
    assert list(X.columns) == PATIENT_FEATURES
    assert not set(X.columns) & set(C.LEAKAGE_REGISTRY)


def test_patients_never_cross_splits(cohort):
    P = cohort.patients
    assert set(P[P.split == "dev"].patient_id).isdisjoint(P[P.split == "test"].patient_id)
    assert (P.split == "test").sum() == 80
    assert P.SCr_mg_dl.max() <= C.MAX_SCR_MG_DL


def test_levels_come_from_measured_profile(cohort):
    r = cohort.regimen_table.query("regimen_type == 'standard'").iloc[0]
    prof = cohort.profile(r.regimen_id)
    lv = sample_levels(prof, regimen_from_row(r))
    assert len(lv) == 2 and lv[0][0] < lv[1][0] <= 72
    assert all(c in set(prof.conc_measured_mg_L.round(6)) for c in np.round([x[1] for x in lv], 6))


def test_regimen_validation():
    with pytest.raises(ValueError):
        Regimen(1000, 2, 3).validate()
