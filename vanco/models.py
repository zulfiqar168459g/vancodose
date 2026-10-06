"""Model definitions.

Architecture
------------
1. A-priori clearance model (tabular ML): bedside covariates -> log(CL).
   AUC24 for *any* regimen = daily dose / CL, so one model serves every regimen and the
   prediction is exactly dose-proportional.
2. Population PK prior: log(V1), log(V2), log(Q) ridge models + residual covariance.
3. Bayesian TDM update (MAP): when measured levels exist, the prior is combined with the
   levels through the two-compartment model to give an individual estimate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import PATIENT_FEATURES, patient_features
from .pk import Regimen, concentration

log = logging.getLogger(__name__)
PK_PARAMS = ["true_CL_L_h", "true_Vd1_L", "true_Vd2_L", "true_Q_L_h"]


# ---------------------------------------------------------------------------
# Candidate a-priori estimators and their hyper-parameter search spaces
# ---------------------------------------------------------------------------
def make_estimator(name: str, params: dict | None = None):
    p = dict(params or {})
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=p.get("alpha", 1.0)))
    if name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=p.get("alpha", 1e-3), max_iter=20000))
    if name == "elastic_net":
        return make_pipeline(StandardScaler(), ElasticNet(alpha=p.get("alpha", 1e-3), l1_ratio=p.get("l1_ratio", 0.5), max_iter=20000))
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=400, random_state=C.SEED, n_jobs=-1, **p)
    if name == "extra_trees":
        return ExtraTreesRegressor(n_estimators=400, random_state=C.SEED, n_jobs=-1, **p)
    if name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=C.SEED, n_jobs=4, verbosity=0, **p)
    if name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=C.SEED, verbose=-1, n_jobs=4, **p)
    if name == "mlp":
        layers = tuple([p.pop("width", 32)] * p.pop("depth", 2))
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=layers, early_stopping=True, max_iter=3000,
                                                            random_state=C.SEED, **p))
    raise ValueError(f"Unknown estimator {name}")


def search_space(name: str, trial) -> dict:
    """Optuna search spaces (bounded to control complexity on 320 patients)."""
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
    if name == "lasso":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True)}
    if name == "elastic_net":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True), "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95)}
    if name in ("random_forest", "extra_trees"):
        return {"max_depth": trial.suggest_int("max_depth", 3, 14), "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_float("max_features", 0.3, 1.0)}
    if name == "xgboost":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=100), "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 6), "min_child_weight": trial.suggest_float("min_child_weight", 1, 20, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0), "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True), "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5, log=True)}
    if name == "lightgbm":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=100), "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 3, 31), "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1, "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True)}
    if name == "mlp":
        return {"width": trial.suggest_categorical("width", [16, 32, 64]), "depth": trial.suggest_int("depth", 1, 3),
                "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True), "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)}
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Population PK prior + Bayesian update
# ---------------------------------------------------------------------------
@dataclass
class PopulationPK:
    """A-priori model for log(CL, V1, V2, Q) and its between-patient covariance."""
    cl_name: str = "ridge"
    cl_params: dict = field(default_factory=dict)
    cl_model: object = None
    aux_model: object = None
    omega: np.ndarray | None = None
    error_model: tuple = C.DEFAULT_ERROR_MODEL

    def fit(self, patients: pd.DataFrame, profiles: pd.DataFrame | None = None) -> "PopulationPK":
        X = patient_features(patients)
        Y = np.log(patients[PK_PARAMS].to_numpy(float))
        groups = patients["patient_id"].to_numpy()
        self.cl_model = make_estimator(self.cl_name, self.cl_params).fit(X, Y[:, 0])
        self.aux_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, Y[:, 1:])
        # Out-of-fold residuals give an honest between-patient covariance.
        oof = np.zeros_like(Y)
        for tr, va in GroupKFold(n_splits=C.N_FOLDS).split(X, groups=groups):
            oof[va, 0] = clone(self.cl_model).fit(X.iloc[tr], Y[tr, 0]).predict(X.iloc[va])
            oof[va, 1:] = clone(self.aux_model).fit(X.iloc[tr], Y[tr, 1:]).predict(X.iloc[va])
        self.omega = np.cov((Y - oof).T)
        if profiles is not None:
            self.error_model = fit_error_model(profiles)
        return self

    def prior(self, X: pd.DataFrame) -> np.ndarray:
        """Prior mean of log(CL, V1, V2, Q); shape (n, 4)."""
        return np.column_stack([self.cl_model.predict(X), self.aux_model.predict(X)])

    @property
    def cl_sd(self) -> float:
        return float(np.sqrt(self.omega[0, 0]))


def fit_error_model(profiles: pd.DataFrame, n: int = 200_000) -> tuple[float, float]:
    """Combined additive (mg/L) + proportional assay error, by maximum likelihood."""
    d = profiles[profiles["conc_true_mg_L"] > 0.5]
    d = d.sample(min(n, len(d)), random_state=C.SEED)
    c = d["conc_true_mg_L"].to_numpy()
    r2 = (d["conc_measured_mg_L"].to_numpy() - c) ** 2
    def nll(q):
        v = q[0] ** 2 + (q[1] * c) ** 2
        return float(np.mean(np.log(v) + r2 / v))
    q = minimize(nll, [1.0, 0.1], method="Nelder-Mead").x
    return (float(abs(q[0])), float(abs(q[1])))


@dataclass
class BayesianEstimate:
    theta: np.ndarray          # MAP log(CL, V1, V2, Q)
    prior_mean: np.ndarray
    log_cl_sd: float           # Laplace posterior SD of log CL
    n_levels: int

    @property
    def params(self) -> tuple[float, float, float, float]:
        return tuple(np.exp(self.theta))


def bayesian_update(prior_mean: np.ndarray, omega: np.ndarray, regimen: Regimen,
                    levels: list[tuple[float, float]], error_model=C.DEFAULT_ERROR_MODEL,
                    uncertainty: bool = True) -> BayesianEstimate:
    """Maximum a-posteriori individual PK parameters given measured levels."""
    if not levels:
        return BayesianEstimate(prior_mean.copy(), prior_mean.copy(), float(np.sqrt(omega[0, 0])), 0)
    oi = np.linalg.inv(omega)
    t = np.array([lv[0] for lv in levels], float)
    y = np.array([lv[1] for lv in levels], float)
    add, prop = error_model

    def nlp(th):
        c = concentration(t, regimen, *np.exp(th))
        v = add ** 2 + (prop * c) ** 2
        d = th - prior_mean
        return 0.5 * np.sum((y - c) ** 2 / v + np.log(v)) + 0.5 * d @ oi @ d

    res = minimize(nlp, prior_mean, method="L-BFGS-B", bounds=[(m - 3, m + 3) for m in prior_mean])
    th = res.x
    if not uncertainty:
        return BayesianEstimate(th, prior_mean.copy(), float("nan"), len(levels))
    # Laplace approximation: numerical Hessian -> posterior SD of log CL.
    h = 1e-3
    H = np.zeros((4, 4))
    f0 = nlp(th)
    for i in range(4):
        for j in range(i, 4):
            ei, ej = np.eye(4)[i] * h, np.eye(4)[j] * h
            H[i, j] = H[j, i] = (nlp(th + ei + ej) - nlp(th + ei) - nlp(th + ej) + f0) / h ** 2
    try:
        cov = np.linalg.inv(H + 1e-9 * np.eye(4))
        sd = float(np.sqrt(max(cov[0, 0], 1e-6)))
    except np.linalg.LinAlgError:
        sd = float(np.sqrt(omega[0, 0]))
    return BayesianEstimate(th, prior_mean.copy(), min(sd, float(np.sqrt(omega[0, 0]))), len(levels))
