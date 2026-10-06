# VancoDose

[![Launcher test](../../actions/workflows/launcher-test.yml/badge.svg)](../../actions/workflows/launcher-test.yml)

Vancomycin exposure prediction and precision dosing, built on the synthetic *Sentinel Vancomycin
Precision-Dosing Cohort* (400 simulated patients; 394 after outlier exclusion).

**Research prototype. Not for clinical use.**

## Quick start

No setup needed. The launcher checks the computer, installs anything missing (Python 3.12,
packages, the model) and opens VancoDose in your browser. The first start takes a few minutes;
later starts take seconds. Keep the launcher window open while you use the app.

| Computer | What to do |
|---|---|
| **macOS** | Double-click **`Start VancoDose.command`**. The first time, macOS may say it is from an unidentified developer: right-click the file, choose **Open**, then **Open** again. If Python is missing, macOS asks for your password once to install it. The file gets the VancoDose icon after the first run. |
| **Windows 10/11** | Double-click **`Start VancoDose.bat`**. If Windows SmartScreen appears, click **More info → Run anyway**. Python is installed for your user account if needed (no administrator rights). A **VancoDose** shortcut with the app icon is added to your Desktop. |
| **Linux** | Run `./Start\ VancoDose.command` in a terminal (install Python 3.12+ with your package manager first). |

What the launcher handles for you:

* **Python**: finds 64-bit Python 3.12–3.14; otherwise installs Python 3.12 (Homebrew or the signed python.org
  installer on macOS; winget or the signed python.org installer on Windows). Installer signatures are verified.
* **Packages**: a private environment in `.venv`, reinstalled only when `requirements.txt` changes. If exact
  versions are not available for the computer, compatible versions from `requirements-compat.txt` are used.
* **Model**: verified at every start and retrained locally (about 30 s) if it is missing or does not match the installed libraries.
* **Port**: uses 8000, or the next free port; if VancoDose is already running it just opens the browser.
* **Problems**: plain-language messages with the fix, and full details in `logs/launcher.log`.

Every update to the GitHub repository automatically re-tests the launchers on fresh Windows, macOS and Linux
machines (`.github/workflows/launcher-test.yml`): Python installation on Windows, package installation, model
check, app start-up and real predictions.

Options (add after the launcher name in a terminal): `--check` installs and verifies only, `--reinstall` rebuilds
the environment, `--port 8500`, `--no-browser`.

Manual setup for developers:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m vanco.experiments --trials 30     # full model comparison (≈1 h)
.venv/bin/python -m vanco.train && .venv/bin/python -m vanco.report
.venv/bin/python -m pytest -q
.venv/bin/python -m uvicorn app.server:app --port 8000
```

## Publishing to GitHub

Double-click **`Publish to GitHub.command`** (macOS; on Windows run `python scripts\publish_github.py`). It signs in to GitHub in
the browser with a separate login used only by this project (an existing business login on the computer is never
used), shows the account and asks you to confirm it is your personal one, lists the files, then creates the
repository and uploads. Commits use your GitHub no-reply e-mail address.

## Using the app

| Tab | What it is for |
|---|---|
| **Search Patient by ID** | Type or pick a patient ID. See every course as a journey (condition → dose → response → reason → next dose), patient-level analytics, the factors behind each dosing decision, and plan the next course from the same page. Added courses and levels are saved locally in `data/patient_history.db`; delete that file to reset all histories. |
| **Dose calculator** | Quick what-if for any patient and regimen, without saving. |
| **Population analytics** | Target attainment, AUC distribution and risk by kidney function for the whole cohort. |
| **Model & evidence** | Model comparison, held-out accuracy against the goal, and limitations. |

Keyboard: `/` search patients, `?` glossary, `Esc` close. A patient page can be linked directly, e.g. `http://127.0.0.1:8000/#patients/VANCO_00028`.

## How it works

```
patient (age, sex, weight, SCr → CrCL, ICU, RRT, ARC)
      │
      ▼
① covariate model (Lasso on log-clearance)  ──► a-priori clearance ± uncertainty
      │                                              │
      │  measured levels? ──► ② Bayesian MAP update with a two-compartment PK model
      ▼                                              ▼
③ AUC24 = daily dose ÷ clearance (any regimen)   individual CL, V1, V2, Q
      │
      ▼
clinical category · recommended regimen (250 mg steps, q8–q48h) · SHAP explanation · 72-h curve
```

* **Patient-level prediction → tabular ML.** Eight algorithms (Ridge, Lasso, Elastic Net, Random
  Forest, Extra Trees, XGBoost, LightGBM, MLP) were tuned with Optuna and compared with grouped
  cross-validation. Linear models won on accuracy, stability, speed and explainability.
* **Concentration–time prediction → mechanistic + Bayesian.** GRU sequence models and a tabular
  LightGBM curve model were compared and were less accurate than the two-compartment model with a
  Bayesian update, which reproduces the simulator exactly given the right parameters.
* **Explainability.** SHAP values on log-clearance become multiplicative effects on exposure
  ("creatinine clearance 43 mL/min raises exposure by 84%"). Together with the level adjustment they
  multiply back exactly to the prediction.

Results, leakage analysis and the full comparison tables are in **[reports/REPORT.md](reports/REPORT.md)**.

## Repository layout

| Path | Purpose |
|---|---|
| `vanco/config.py` | Paths, feature groups, leakage registry, outlier rule, clinical targets |
| `vanco/data.py` | Load + join the three tables, outlier exclusion, data audit |
| `vanco/features.py` | Bedside feature engineering, TDM level sampling |
| `vanco/pk.py` | Two-compartment infusion model, steady state |
| `vanco/models.py` | Candidate estimators + search spaces, population PK prior, Bayesian update |
| `vanco/sequence.py` | GRU models (comparison only) |
| `vanco/experiments.py` | Systematic model comparison (development data only) |
| `vanco/train.py` | Selection, one-time test evaluation, production refit, figures |
| `vanco/evaluate.py` | Metrics and clinical interpretation |
| `vanco/explain.py` | SHAP explanations in clinical language |
| `vanco/predict.py` | Production inference and regimen recommendation |
| `vanco/history.py` | Patient treatment history: storage, multi-course Bayesian timeline, reasons, flags |
| `vanco/analytics.py` | Population analytics for the dashboard |
| `vanco/report.py` | Builds `reports/REPORT.md` |
| `app/` | FastAPI server + single-page web app (no build step) |
| `Start VancoDose.command` / `.bat` | One-click launchers for macOS/Linux and Windows |
| `scripts/launcher.py` | Cross-platform start-up: checks, installs, model, server |
| `scripts/start-vancodose.ps1`, `scripts/common.sh` | Python detection and installation per platform |
| `scripts/publish_github.py` | Publishes the project to a personal GitHub account |
| `assets/` | VancoDose icon (PNG, Windows ICO) |
| `models/` | Persisted model bundle and model card |
| `reports/` | Comparison tables, test evaluation, figures, report |
| `Reference Data/` | Background material supplied with the project |
| `data/` | Original Parquet files (unchanged) and the patient split |
| `tests/` | Data, PK, leakage, prediction and API tests |

## Limits

* Synthetic data only; 9 patients on renal replacement therapy. Real-world validation is required.
* Patients with serum creatinine > 3.5 mg/dL were excluded as outliers, so they are outside the model's scope.
* AUC24 targets assume MIC = 1 mg/L (2020 ASHP/IDSA/PIDS/SIDP consensus guideline).
* Accuracy goal: MAE < 90 and RMSE < 100 mg·h/L. Met for the regimen VancoDose recommends after a peak and trough
  (test MAE 72, RMSE 91); MAE met but RMSE not (≈104–106) for regimens aimed at the target range; not met across all
  regimens, whose simulated AUC24 reaches several thousand. See section 6 of the report.
* Follow-up levels shown as "simulated" exist only for these synthetic patients; real use requires measured levels.
* Across courses the Bayesian update assumes clearance is stable; when creatinine changes, the prior is updated but
  earlier levels still count fully.
