"""Select, evaluate once on the held-out test patients, refit on all data and persist.

Run:  python -m vanco.train
Requires reports/apriori_comparison.csv etc. from ``python -m vanco.experiments``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402
from .data import audit, load_cohort  # noqa: E402
from .evaluate import by_band, category_agreement, regression_metrics  # noqa: E402
from .explain import ClearanceExplainer  # noqa: E402
from .features import patient_features, regimen_from_row, sample_levels  # noqa: E402
from .models import BayesianEstimate, PopulationPK, bayesian_update  # noqa: E402
from .pk import Regimen, concentration, steady_state  # noqa: E402
from .predict import Bundle, VancoPredictor  # noqa: E402

log = logging.getLogger("vanco.train")
PALETTE = {"purple": "#B36CFF", "violet": "#844BFF", "ultra": "#5E6BFF", "river": "#43C0FF", "grey": "#8A8FA3", "ink": "#1E1B3A"}
LINEAR = {"ridge", "lasso", "elastic_net"}


def select_apriori(df: pd.DataFrame, tolerance=0.02) -> str:
    """Lowest CV MAE; any linear model within 2% wins (simpler, exact explanations, portable)."""
    cand = df[~df.model.str.startswith("baseline")].sort_values("CV_MAE")
    best = cand.iloc[0]
    close = cand[(cand.CV_MAE <= best.CV_MAE * (1 + tolerance)) & cand.model.isin(LINEAR)]
    return (close.iloc[0] if len(close) else best).model


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=PALETTE["ink"])
    ax.grid(axis="y", color="#ECEAF5", lw=0.8)
    ax.set_axisbelow(True)


def evaluate_test(cohort, pop: PopulationPK) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """One-time evaluation on the 80 held-out patients."""
    P, R = cohort.patients, cohort.regimen_table
    test_reg = R[R.patient_id.isin(P[P.split == "test"].patient_id)].reset_index(drop=True)
    X = patient_features(test_reg)
    prior = pop.prior(X)
    rows = []
    for i, r in test_reg.iterrows():
        g = regimen_from_row(r)
        row = {"patient_id": r.patient_id, "type": r.regimen_type, "true_auc": r.AUC24_true_mg_h_L, "daily": r.daily_dose_mg,
               "true_cl": r.true_CL_L_h, "prior_lcl": prior[i, 0], "prior_theta": prior[i].tolist(), "true_peak": r.peak_ss_mg_L, "true_trough": r.trough_ss_mg_L}
        if r.regimen_type != "auc_targeted":  # levels are drawn from the regimen actually given
            prof = cohort.profile(r.regimen_id)
            for d in ("trough", "peak_trough", "two_pairs"):
                e = bayesian_update(prior[i], pop.omega, g, sample_levels(prof, g, d), pop.error_model)
                row[f"{d}_lcl"], row[f"{d}_sd"] = e.theta[0], e.log_cl_sd
                if d == "peak_trough":
                    row["pt_theta"] = e.theta.tolist()
                    ss = steady_state(g, *e.params)
                    row["pred_peak"], row["pred_trough"] = ss["peak_ss_mg_L"], ss["trough_ss_mg_L"]
                    t = np.arange(1, 72.0001, 1.0)
                    row["curve_mae"] = float(np.mean(np.abs(concentration(t, g, *e.params) - np.interp(t, prof.time_h, prof.conc_true_mg_L))))
            row["curve_mae_prior"] = float(np.mean(np.abs(concentration(np.arange(1, 72.0001, 1.0), g, *np.exp(prior[i])) -
                                                          np.interp(np.arange(1, 72.0001, 1.0), prof.time_h, prof.conc_true_mg_L))))
        rows.append(row)
    T = pd.DataFrame(rows)
    # Proposed (auc_targeted) regimens: clearance estimated from the patient's standard-regimen levels.
    std = T[T.type == "standard"].set_index("patient_id")
    tgt = T[T.type == "auc_targeted"].copy()
    tgt["peak_trough_lcl"] = tgt.patient_id.map(std.peak_trough_lcl)
    tgt["peak_trough_sd"] = tgt.patient_id.map(std.peak_trough_sd)
    tgt["two_pairs_lcl"] = tgt.patient_id.map(std.two_pairs_lcl)
    given = T[T.type != "auc_targeted"]

    def m(df, col):
        p = df.daily / np.exp(df[col])
        return {**regression_metrics(df.true_auc, p), "category_agreement": category_agreement(df.true_auc, p)}

    res = {
        "a_priori_given_regimens": m(given, "prior_lcl"),
        "bayes_trough_given_regimens": m(given, "trough_lcl"),
        "bayes_peak_trough_given_regimens": m(given, "peak_trough_lcl"),
        "a_priori_target_range_regimens": m(tgt, "prior_lcl"),
        "bayes_peak_trough_target_range_regimens": m(tgt, "peak_trough_lcl"),
        "bayes_two_pairs_given_regimens": m(given, "two_pairs_lcl"),
        "bayes_two_pairs_target_range_regimens": m(tgt, "two_pairs_lcl"),
        "goal": {"MAE": C.GOAL_MAE, "RMSE": C.GOAL_RMSE},
        "steady_state_peak_bayes": regression_metrics(given.true_peak, given.pred_peak),
        "steady_state_trough_bayes": regression_metrics(given.true_trough, given.pred_trough),
        "curve_MAE_mg_L_bayes": float(given.curve_mae.mean()),
        "curve_MAE_mg_L_prior": float(given.curve_mae_prior.mean()),
    }
    # 80% interval coverage (Laplace posterior) for the Bayesian AUC
    lo = given.daily / np.exp(given.peak_trough_lcl + 1.2816 * given.peak_trough_sd)
    hi = given.daily / np.exp(given.peak_trough_lcl - 1.2816 * given.peak_trough_sd)
    res["bayes_80pct_interval_coverage"] = float(np.mean((given.true_auc >= lo) & (given.true_auc <= hi)))
    lo = given.daily / np.exp(given.prior_lcl + 1.2816 * pop.cl_sd)
    hi = given.daily / np.exp(given.prior_lcl - 1.2816 * pop.cl_sd)
    res["a_priori_80pct_interval_coverage"] = float(np.mean((given.true_auc >= lo) & (given.true_auc <= hi)))
    # Dosing simulation: recommended regimen evaluated against the latent truth
    res["target_attainment"] = _attainment(T, pop, cohort)
    bands = pd.concat([by_band(given.true_auc, given.daily / np.exp(given.prior_lcl)).assign(model="a-priori"),
                       by_band(given.true_auc, given.daily / np.exp(given.peak_trough_lcl)).assign(model="Bayesian peak+trough")])
    return res, T, bands


def _attainment(T, pop, cohort) -> dict:
    P = cohort.patients.set_index("patient_id")
    std = T[T.type == "standard"].set_index("patient_id")
    predictor = VancoPredictor(Bundle(pop, None, {}, {}))
    out = {"standard_dosing": [], "recommended_a_priori": [], "recommended_after_levels": [], "dataset_auc_targeted": []}
    pred = {"recommended_a_priori": [], "recommended_after_levels": []}
    for pid, r in std.iterrows():
        out["standard_dosing"].append(r.true_auc)
        tgt = T[(T.patient_id == pid) & (T.type == "auc_targeted")].true_auc
        out["dataset_auc_targeted"].append(float(tgt.iloc[0]))
        for key, theta, sd, n in (("recommended_a_priori", r.prior_theta, pop.cl_sd, 0),
                                  ("recommended_after_levels", r.pt_theta, r.peak_trough_sd, 2)):
            est = BayesianEstimate(np.asarray(theta), np.asarray(r.prior_theta), float(sd), n)
            best = predictor.recommend(est, P.loc[pid].to_dict())["best"]
            out[key].append(best["daily_dose_mg"] / P.loc[pid].true_CL_L_h)
            pred[key].append(best["auc24"])
    summary = {}
    for k in pred:  # how accurate is the AUC the tool promises for its own recommendation?
        summary[f"{k}_prediction_error"] = regression_metrics(np.array(out[k]), np.array(pred[k]))
    for k, v in out.items():
        v = np.array(v)
        summary[k] = {"pct_within_400_600": float(np.mean((v >= 400) & (v <= 600)) * 100), "pct_below_400": float(np.mean(v < 400) * 100),
                      "pct_above_600": float(np.mean(v > 600) * 100), "pct_above_800": float(np.mean(v > 800) * 100),
                      "median_auc": float(np.median(v))}
    summary["_note"] = ("dataset_auc_targeted was designed by the simulator using the latent clearance (an oracle-informed "
                        "reference, not achievable in practice).")
    return summary


def figures(cohort, apriori, T, pop, explainer, X_all):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # 1. a-priori comparison
    d = apriori.sort_values("CV_MAE")
    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d.CV_MAE, 0.4, color=PALETTE["violet"], label="MAE")
    ax.barh(y + 0.2, d.CV_RMSE, 0.4, color=PALETTE["river"], label="RMSE")
    ax.set_yticks(y, d.model.str.replace("_", " "))
    ax.axvline(70, color=PALETTE["grey"], ls="--", lw=1)
    ax.text(72, len(d) - 0.6, "70 target", color=PALETTE["grey"], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("AUC24 error (mg·h/L), 5-fold patient CV ×3"); ax.legend(frameon=False)
    ax.set_title("Covariate-only models: all algorithms hit the same information ceiling", fontsize=10, loc="left")
    _style(ax); fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "apriori_comparison.png", dpi=150); plt.close(fig)
    # 2. predicted vs true
    given = T[T.type != "auc_targeted"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2), sharex=True, sharey=True)
    for ax, col, title, colr in ((axs[0], "prior_lcl", "Covariates only", PALETTE["purple"]),
                                 (axs[1], "peak_trough_lcl", "Covariates + peak & trough", PALETTE["ultra"])):
        p = given.daily / np.exp(given[col])
        ax.scatter(given.true_auc, p, s=14, color=colr, alpha=0.75, edgecolor="none")
        ax.plot([50, 15000], [50, 15000], color=PALETTE["grey"], lw=1)
        ax.axhspan(400, 600, color=PALETTE["river"], alpha=0.10)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("True AUC24 (mg·h/L)"); _style(ax)
    axs[0].set_ylabel("Predicted AUC24 (mg·h/L)")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "test_pred_vs_true.png", dpi=150); plt.close(fig)
    # 3. SHAP summary (mean |effect| on exposure)
    phi = explainer.shap_values(X_all)
    imp = pd.Series(np.mean(np.abs(phi), 0), index=X_all.columns).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([c.replace("_", " ") for c in imp.index], (np.exp(imp.values) - 1) * 100, color=PALETTE["violet"])
    ax.set_xlabel("Mean absolute effect on predicted exposure (%)"); ax.set_title("What drives the clearance model", fontsize=10, loc="left")
    _style(ax); fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "feature_importance.png", dpi=150); plt.close(fig)


def write_audit(a: dict):
    (C.REPORTS_DIR / "data_audit.json").write_text(json.dumps(a, indent=2, default=str))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cohort = load_cohort()
    write_audit(audit(cohort))
    apriori = pd.read_csv(C.REPORTS_DIR / "apriori_comparison.csv")
    best = json.loads((C.REPORTS_DIR / "best_params.json").read_text())
    name = select_apriori(apriori)
    selection = {"apriori_model": name, "params": best.get(name, {}), "tdm_method": "bayesian_map",
                 "curve_method": "mechanistic_two_compartment", "rule": "lowest CV MAE; linear model preferred within 2%"}
    (C.REPORTS_DIR / "selection.json").write_text(json.dumps(selection, indent=2))
    log.info("Selected a-priori model: %s %s", name, selection["params"])

    P = cohort.patients
    profiles = cohort.profiles.reset_index()
    dev = P[P.split == "dev"]
    pop_dev = PopulationPK(name, selection["params"]).fit(dev, profiles[profiles.patient_id.isin(dev.patient_id)])
    res, T, bands = evaluate_test(cohort, pop_dev)
    (C.REPORTS_DIR / "test_evaluation.json").write_text(json.dumps(res, indent=2))
    T.to_csv(C.REPORTS_DIR / "test_predictions.csv", index=False)
    bands.to_csv(C.REPORTS_DIR / "test_error_by_auc_band.csv", index=False)
    log.info("Test: a-priori MAE %.1f | Bayesian peak+trough MAE %.1f | target-range regimens MAE %.1f",
             res["a_priori_given_regimens"]["MAE"], res["bayes_peak_trough_given_regimens"]["MAE"],
             res["bayes_peak_trough_target_range_regimens"]["MAE"])

    # Production refit on all 400 patients (selection and hyper-parameters frozen above).
    pop = PopulationPK(name, selection["params"]).fit(P, profiles)
    X_all = patient_features(P)
    explainer = ClearanceExplainer(pop.cl_model, X_all)
    figures(cohort, apriori, T, pop, explainer, X_all)
    ranges = {k: (float(P[k].min()), float(P[k].max())) for k in ["age", "weight_kg", "height_cm", "SCr_mg_dl", "CrCL_ml_min"]}
    card = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "selection": selection,
            "training_patients": int(len(P)), "error_model": pop.error_model, "omega_sd": np.sqrt(np.diag(pop.omega)).tolist(),
            "test_metrics": res, "features": list(X_all.columns),
            "sklearn_version": __import__("sklearn").__version__, "python_version": __import__("platform").python_version()}
    C.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(Bundle(pop, explainer, ranges, card), C.MODEL_FILE)
    C.MODEL_CARD.write_text(json.dumps(card, indent=2, default=float))
    log.info("Saved %s", C.MODEL_FILE)


if __name__ == "__main__":
    main()
