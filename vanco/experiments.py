"""Systematic model comparison (development data only, grouped by patient).

Run:  python -m vanco.experiments [--trials 30]

Outputs (reports/): apriori_comparison.csv, tdm_comparison.csv, curve_comparison.csv,
best_params.json. The 80 test patients are never touched here.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from . import config as C
from .data import load_cohort
from .evaluate import category_agreement, regression_metrics
from .features import level_features, patient_features, regimen_from_row, sample_levels
from .models import PopulationPK, bayesian_update, make_estimator, search_space
from .pk import concentration
from .sequence import GRID, GRUClearance, GRUCurve, rate_channel, tdm_sequence

log = logging.getLogger("vanco.experiments")
CANDIDATES = ["ridge", "lasso", "elastic_net", "random_forest", "extra_trees", "xgboost", "lightgbm", "mlp"]
COMPLEXITY = {"ridge": "6 coefficients", "lasso": "<=6 coefficients", "elastic_net": "<=6 coefficients",
              "random_forest": "400 trees", "extra_trees": "400 trees", "xgboost": "boosted trees",
              "lightgbm": "boosted trees", "mlp": "neural network"}
EXPLAIN = {"ridge": "exact (linear)", "lasso": "exact (linear)", "elastic_net": "exact (linear)",
           "random_forest": "TreeSHAP", "extra_trees": "TreeSHAP", "xgboost": "TreeSHAP", "lightgbm": "TreeSHAP", "mlp": "KernelSHAP (approx.)"}


def _dev(cohort):
    P = cohort.patients
    dev = P[P.split == "dev"].reset_index(drop=True)
    R = cohort.regimen_table
    reg = R[R.patient_id.isin(dev.patient_id) & (R.regimen_type != "auc_targeted")].reset_index(drop=True)
    return dev, reg


def _auc_eval(pids, log_cl, reg):
    cl = pd.Series(np.exp(log_cl), index=pids)
    r = reg[reg.patient_id.isin(pids)]
    return r.AUC24_true_mg_h_L.to_numpy(), (r.daily_dose_mg / r.patient_id.map(cl)).to_numpy()


# ---------------------------------------------------------------------------
# A. A-priori (covariate-only) clearance models
# ---------------------------------------------------------------------------
def tune(name, X, y, n_trials):
    if n_trials == 0:
        return {}
    kf = KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED)

    def objective(trial):
        params = search_space(name, trial)
        err = []
        for tr, va in kf.split(X):
            m = make_estimator(name, dict(params)).fit(X.iloc[tr], y[tr])
            err.append(np.mean((m.predict(X.iloc[va]) - y[va]) ** 2))
        return float(np.mean(err))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def run_apriori(cohort, n_trials=30, repeats=3):
    dev, reg = _dev(cohort)
    X, y, pids = patient_features(dev), np.log(dev.true_CL_L_h.to_numpy()), dev.patient_id.to_numpy()
    rows, best = [], {}
    for name in CANDIDATES:
        t0 = time.time()
        params = tune(name, X, y, 0 if name == "ridge" and n_trials == 0 else n_trials)
        if name in ("lightgbm",):
            params = {**params, "subsample_freq": 1}
        best[name] = params
        folds = []
        for rep in range(repeats):  # repeated CV with seeds different from tuning
            for tr, va in KFold(C.N_FOLDS, shuffle=True, random_state=1000 + rep).split(X):
                m = make_estimator(name, dict(params)).fit(X.iloc[tr], y[tr])
                yt, yp = _auc_eval(pids[va], m.predict(X.iloc[va]), reg)
                tt, tp = _auc_eval(pids[tr], m.predict(X.iloc[tr]), reg)
                v, t = regression_metrics(yt, yp), regression_metrics(tt, tp)
                folds.append({"MAE": v["MAE"], "RMSE": v["RMSE"], "R2": v["R2"], "MdAPE": v["MdAPE_%"],
                              "logCL_RMSE": float(np.sqrt(np.mean((m.predict(X.iloc[va]) - y[va]) ** 2))),
                              "train_MAE": t["MAE"], "train_RMSE": t["RMSE"], "cat": category_agreement(yt, yp)})
        f = pd.DataFrame(folds)
        m = make_estimator(name, dict(params)).fit(X, y)
        big = pd.concat([X] * 30, ignore_index=True)
        t1 = time.perf_counter(); m.predict(big); ms = (time.perf_counter() - t1) / len(big) * 1e6
        rows.append({"model": name, "CV_MAE": f.MAE.mean(), "CV_RMSE": f.RMSE.mean(), "CV_R2": f.R2.mean(),
                     "CV_MdAPE_%": f.MdAPE.mean(), "CV_logCL_RMSE": f.logCL_RMSE.mean(), "MAE_fold_SD": f.MAE.std(),
                     "train_MAE": f.train_MAE.mean(), "train_RMSE": f.train_RMSE.mean(),
                     "overfit_ratio_RMSE": f.RMSE.mean() / f.train_RMSE.mean(), "category_agreement": f.cat.mean(),
                     "inference_us_per_patient": ms, "complexity": COMPLEXITY[name], "explainability": EXPLAIN[name],
                     "tuning_s": time.time() - t0})
        log.info("A-priori %-14s CV MAE %.1f RMSE %.1f", name, rows[-1]["CV_MAE"], rows[-1]["CV_RMSE"])
    # Baselines: population median clearance; direct AUC regression (no clearance structure)
    folds = []
    for tr, va in KFold(C.N_FOLDS, shuffle=True, random_state=1000).split(X):
        yt, yp = _auc_eval(pids[va], np.full(len(va), np.median(y[tr])), reg)
        folds.append(regression_metrics(yt, yp))
    rows.append({"model": "baseline_population_median", "CV_MAE": np.mean([f["MAE"] for f in folds]),
                 "CV_RMSE": np.mean([f["RMSE"] for f in folds]), "CV_R2": np.mean([f["R2"] for f in folds]),
                 "complexity": "1 number", "explainability": "trivial"})
    rows.append(_direct_auc_baseline(dev, reg))
    return pd.DataFrame(rows), best


def _direct_auc_baseline(dev, reg):
    """LightGBM predicting AUC directly from covariates + regimen (what earlier versions did)."""
    from lightgbm import LGBMRegressor
    F = patient_features(reg)
    F["log_daily_dose"] = np.log(reg.daily_dose_mg)
    F["interval"], F["infusion"] = reg.interval_h, reg.infusion_duration_h
    pids = dev.patient_id.to_numpy()
    folds = []
    for tr, va in KFold(C.N_FOLDS, shuffle=True, random_state=1000).split(pids):
        mtr, mva = reg.patient_id.isin(pids[tr]), reg.patient_id.isin(pids[va])
        m = LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, verbose=-1, random_state=C.SEED)
        m.fit(F[mtr], reg.AUC24_true_mg_h_L[mtr])
        folds.append(regression_metrics(reg.AUC24_true_mg_h_L[mva], m.predict(F[mva])))
    return {"model": "baseline_direct_AUC_lightgbm", "CV_MAE": np.mean([f["MAE"] for f in folds]),
            "CV_RMSE": np.mean([f["RMSE"] for f in folds]), "CV_R2": np.mean([f["R2"] for f in folds]),
            "complexity": "boosted trees", "explainability": "TreeSHAP"}


# ---------------------------------------------------------------------------
# B. TDM-informed clearance (covariates + measured levels)
# ---------------------------------------------------------------------------
def _tdm_rows(cohort, reg, pop, design, rng, draws, with_map=True):
    rows, seqs = [], []
    X = patient_features(reg)
    prior = pop.prior(X)
    for i, r in reg.iterrows():
        g = regimen_from_row(r)
        prof = cohort.profile(r.regimen_id)
        for _ in range(draws):
            lv = sample_levels(prof, g, design, rng)
            row = {"patient_id": r.patient_id, "y": np.log(r.true_CL_L_h), "daily": r.daily_dose_mg,
                   "auc": r.AUC24_true_mg_h_L, "prior": prior[i, 0], **X.loc[i].to_dict(), **level_features(lv, g)}
            if with_map:
                row["map"] = bayesian_update(prior[i], pop.omega, g, lv, pop.error_model, uncertainty=False).theta[0]
            rows.append(row)
            seqs.append(tdm_sequence(g, lv))
    return pd.DataFrame(rows), np.stack(seqs)


def run_tdm(cohort, cl_name, cl_params, designs=("trough", "peak_trough"), draws=3):
    from lightgbm import LGBMRegressor
    dev, reg = _dev(cohort)
    profiles = cohort.profiles.reset_index()
    pids = dev.patient_id.to_numpy()
    out = []
    for design in designs:
        preds = {k: [] for k in ["A-priori (covariates only)", "Bayesian MAP (PK model + levels)", "Ridge on level features",
                                 "LightGBM on level features", "Hybrid: MAP + LightGBM residual", "GRU on TDM sequence"]}
        truth = []
        for fold, (tr, va) in enumerate(KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED).split(pids)):
            ptr = dev[dev.patient_id.isin(pids[tr])]
            pop = PopulationPK(cl_name, cl_params).fit(ptr, profiles[profiles.patient_id.isin(ptr.patient_id)])
            rtr = reg[reg.patient_id.isin(pids[tr])].reset_index(drop=True)
            rva = reg[reg.patient_id.isin(pids[va])].reset_index(drop=True)
            Dtr, Str = _tdm_rows(cohort, rtr, pop, design, np.random.default_rng(fold), draws)
            Dva, Sva = _tdm_rows(cohort, rva, pop, design, None, 1)
            feats = [c for c in Dtr.columns if c not in ("patient_id", "y", "daily", "auc", "map")]
            feats = [c for c in feats if Dtr[c].notna().all()]
            lin = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Dtr[feats + ["map"]], Dtr.y)
            gbm = LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=15, min_child_samples=20, subsample=0.8,
                                subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=C.SEED).fit(Dtr[feats], Dtr.y)
            res = LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=7, min_child_samples=40, verbose=-1,
                                random_state=C.SEED).fit(Dtr[feats + ["map"]], Dtr.y - Dtr["map"])
            static_cols = [c for c in feats if not c.startswith(("log_peak", "log_trough", "ke_sz", "t_trough"))]
            gru = GRUClearance().fit(Str, Dtr[static_cols].to_numpy(np.float32), Dtr.y.to_numpy(np.float32))
            preds["A-priori (covariates only)"].append(Dva.prior)
            preds["Bayesian MAP (PK model + levels)"].append(Dva["map"])
            preds["Ridge on level features"].append(lin.predict(Dva[feats + ["map"]]))
            preds["LightGBM on level features"].append(gbm.predict(Dva[feats]))
            preds["Hybrid: MAP + LightGBM residual"].append(Dva["map"] + res.predict(Dva[feats + ["map"]]))
            preds["GRU on TDM sequence"].append(gru.predict(Sva, Dva[static_cols].to_numpy(np.float32)))
            truth.append(Dva[["y", "daily", "auc"]])
            log.info("TDM %s fold %d done", design, fold)
        T = pd.concat(truth, ignore_index=True)
        for name, p in preds.items():
            lc = np.concatenate([np.asarray(x) for x in p])
            m = regression_metrics(T.auc, T.daily / np.exp(lc))
            out.append({"design": design, "model": name, "CV_MAE": m["MAE"], "CV_RMSE": m["RMSE"], "CV_R2": m["R2"],
                        "CV_MdAPE_%": m["MdAPE_%"], "CV_logCL_RMSE": float(np.sqrt(np.mean((lc - T.y) ** 2))),
                        "category_agreement": category_agreement(T.auc, T.daily / np.exp(lc))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# C. Concentration-time curves
# ---------------------------------------------------------------------------
def _curve_tab(reg, X, grid):
    """Tabular features for a direct curve model: patient x regimen x time."""
    rows = []
    for i, r in reg.iterrows():
        g = regimen_from_row(r)
        ev = g.events()
        starts = np.array([e[0] for e in ev])
        for t in grid:
            k = int(np.searchsorted(starts, t, side="right") - 1)
            rows.append({**X.loc[i].to_dict(), "t": t, "since_dose": t - starts[max(k, 0)], "dose_no": k,
                         "dose": g.dose_mg, "interval": g.interval_h, "infusion": g.infusion_duration_h,
                         "loading": g.loading_dose_mg, "cum_dose": sum(e[1] for e in ev if e[0] <= t)})
    return pd.DataFrame(rows)


def run_curves(cohort, cl_name, cl_params):
    from lightgbm import LGBMRegressor
    dev, reg = _dev(cohort)
    profiles = cohort.profiles.reset_index()
    pids = dev.patient_id.to_numpy()
    eval_grid = np.arange(1, 72 + 1e-9, 1.0)
    res = {k: [] for k in ["Mechanistic, a-priori parameters", "Mechanistic, Bayesian (2 levels)",
                           "LightGBM tabular curve", "GRU sequence curve"]}
    truth, meas = [], []
    for fold, (tr, va) in enumerate(KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED).split(pids)):
        ptr = dev[dev.patient_id.isin(pids[tr])]
        pop = PopulationPK(cl_name, cl_params).fit(ptr, profiles[profiles.patient_id.isin(ptr.patient_id)])
        rtr = reg[reg.patient_id.isin(pids[tr])].reset_index(drop=True)
        rva = reg[reg.patient_id.isin(pids[va])].reset_index(drop=True)
        Xtr, Xva = patient_features(rtr), patient_features(rva)
        # Direct ML models are trained on *measured* concentrations (what a lab would provide).
        tab = _curve_tab(rtr, Xtr, eval_grid)
        ytab = np.concatenate([np.interp(eval_grid, cohort.profile(r).time_h, cohort.profile(r).conc_measured_mg_L) for r in rtr.regimen_id])
        gbm = LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=30, verbose=-1, random_state=C.SEED).fit(tab, ytab)
        seq_tr = np.stack([rate_channel(regimen_from_row(r))[:, None] for _, r in rtr.iterrows()]).astype(np.float32)
        ytr = np.stack([np.interp(GRID, cohort.profile(r).time_h, cohort.profile(r).conc_measured_mg_L) for r in rtr.regimen_id]).astype(np.float32)
        static_tr = np.column_stack([Xtr.to_numpy(), rtr[["dose_mg", "interval_h", "infusion_duration_h", "loading_dose_mg"]].to_numpy()]).astype(np.float32)
        gru = GRUCurve().fit(seq_tr, static_tr, ytr)
        seq_va = np.stack([rate_channel(regimen_from_row(r))[:, None] for _, r in rva.iterrows()]).astype(np.float32)
        static_va = np.column_stack([Xva.to_numpy(), rva[["dose_mg", "interval_h", "infusion_duration_h", "loading_dose_mg"]].to_numpy()]).astype(np.float32)
        gru_pred = gru.predict(seq_va, static_va)
        gbm_pred = gbm.predict(_curve_tab(rva, Xva, eval_grid)).reshape(len(rva), -1)
        prior = pop.prior(Xva)
        for i, r in rva.iterrows():
            g = regimen_from_row(r)
            prof = cohort.profile(r.regimen_id)
            truth.append(np.interp(eval_grid, prof.time_h, prof.conc_true_mg_L))
            meas.append(np.interp(eval_grid, prof.time_h, prof.conc_measured_mg_L))
            res["Mechanistic, a-priori parameters"].append(concentration(eval_grid, g, *np.exp(prior[i])))
            est = bayesian_update(prior[i], pop.omega, g, sample_levels(prof, g), pop.error_model, uncertainty=False)
            res["Mechanistic, Bayesian (2 levels)"].append(concentration(eval_grid, g, *est.params))
            res["LightGBM tabular curve"].append(gbm_pred[i])
            res["GRU sequence curve"].append(np.interp(eval_grid, GRID, gru_pred[i]))
        log.info("Curves fold %d done", fold)
    T, Mz = np.concatenate(truth), np.concatenate(meas)
    out = []
    for k, v in res.items():
        p = np.concatenate(v)
        out.append({"model": k, "MAE_vs_true_mg_L": float(np.mean(np.abs(p - T))), "RMSE_vs_true_mg_L": float(np.sqrt(np.mean((p - T) ** 2))),
                    "MAE_vs_measured_mg_L": float(np.mean(np.abs(p - Mz))), "R2_vs_true": regression_metrics(T, p)["R2"]})
    return pd.DataFrame(out)


def run_sampling_designs(cohort, cl_name, cl_params, designs=("trough", "peak_trough", "two_pairs")):
    """Bayesian MAP accuracy by TDM sampling design (grouped CV, development patients only).

    Reports both the regimen the levels came from and the patient's auc_targeted regimen
    (a regimen aimed at the target range, predicted from the standard-regimen levels).
    """
    dev, reg = _dev(cohort)
    R = cohort.regimen_table
    tgt = R[(R.regimen_type == "auc_targeted") & R.patient_id.isin(dev.patient_id)].set_index("patient_id")
    profiles = cohort.profiles.reset_index()
    pids = dev.patient_id.to_numpy()
    rows = {d: [] for d in designs}
    for tr, va in KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED).split(pids):
        ptr = dev[dev.patient_id.isin(pids[tr])]
        pop = PopulationPK(cl_name, cl_params).fit(ptr, profiles[profiles.patient_id.isin(ptr.patient_id)])
        rva = reg[reg.patient_id.isin(pids[va])].reset_index(drop=True)
        prior = pop.prior(patient_features(rva))
        for i, r in rva.iterrows():
            g = regimen_from_row(r)
            for d in designs:
                e = bayesian_update(prior[i], pop.omega, g, sample_levels(cohort.profile(r.regimen_id), g, d), pop.error_model, uncertainty=False)
                rows[d].append((r.regimen_type, r.patient_id, r.daily_dose_mg, r.AUC24_true_mg_h_L, e.theta[0]))
    out = []
    for d, v in rows.items():
        df = pd.DataFrame(v, columns=["type", "pid", "daily", "auc", "lcl"])
        m = regression_metrics(df.auc, df.daily / np.exp(df.lcl))
        st = df[df.type == "standard"].set_index("pid")
        t = tgt.loc[st.index]
        mt = regression_metrics(t.AUC24_true_mg_h_L, t.daily_dose_mg / np.exp(st.lcl))
        out.append({"design": d, "n_levels": {"trough": 1, "peak_trough": 2, "two_pairs": 4}[d],
                    "given_MAE": m["MAE"], "given_RMSE": m["RMSE"], "target_range_MAE": mt["MAE"], "target_range_RMSE": mt["RMSE"]})
    return pd.DataFrame(out)


def run_ablation(cohort, name="elastic_net", params=None):
    """Drop-one feature ablation on the full candidate set (repeated grouped CV, log-CL RMSE)."""
    from .features import CANDIDATE_FEATURES, PATIENT_FEATURES
    dev, _ = _dev(cohort)
    X, y = patient_features(dev, CANDIDATE_FEATURES), np.log(dev.true_CL_L_h.to_numpy())

    def cv(cols):
        e = []
        for rep in range(3):
            for tr, va in KFold(C.N_FOLDS, shuffle=True, random_state=1000 + rep).split(X):
                m = make_estimator(name, dict(params or {})).fit(X.iloc[tr][cols], y[tr])
                e.append(np.sqrt(np.mean((m.predict(X.iloc[va][cols]) - y[va]) ** 2)))
        return float(np.mean(e))
    full = cv(CANDIDATE_FEATURES)
    rows = [{"feature_set": "all 11 candidates", "logCL_RMSE": full}]
    rows += [{"feature_set": f"drop {f}", "logCL_RMSE": cv([c for c in CANDIDATE_FEATURES if c != f])} for f in CANDIDATE_FEATURES]
    rows.append({"feature_set": "final six: " + ", ".join(PATIENT_FEATURES), "logCL_RMSE": cv(PATIENT_FEATURES)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    C.REPORTS_DIR.mkdir(exist_ok=True)
    cohort = load_cohort()
    best_file = C.REPORTS_DIR / "best_params.json"
    if "apriori" not in a.skip:
        df, best = run_apriori(cohort, a.trials)
        df.to_csv(C.REPORTS_DIR / "apriori_comparison.csv", index=False)
        best_file.write_text(json.dumps(best, indent=2))
    best = json.loads(best_file.read_text())
    from .train import select_apriori
    name = select_apriori(pd.read_csv(C.REPORTS_DIR / "apriori_comparison.csv"))
    log.info("A-priori model used for TDM/curve comparisons: %s", name)
    if "ablation" not in a.skip:
        run_ablation(cohort, "elastic_net", best.get("elastic_net")).to_csv(C.REPORTS_DIR / "feature_ablation.csv", index=False)
    if "designs" not in a.skip:
        run_sampling_designs(cohort, name, best.get(name, {})).to_csv(C.REPORTS_DIR / "sampling_designs.csv", index=False)
    if "tdm" not in a.skip:
        run_tdm(cohort, name, best.get(name, {})).to_csv(C.REPORTS_DIR / "tdm_comparison.csv", index=False)
    if "curves" not in a.skip:
        run_curves(cohort, name, best.get(name, {})).to_csv(C.REPORTS_DIR / "curve_comparison.csv", index=False)


if __name__ == "__main__":
    main()
