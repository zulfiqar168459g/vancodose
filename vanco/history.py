"""Patient-level treatment history: storage, timeline reasoning and per-course explanations.

Course 1 of every patient comes from the dataset (its standard regimen and a protocol peak/trough
drawn from the measured profile). Further courses and levels are added by the user and stored in a
local SQLite file. Each course is explained in the order a clinician thinks:

    patient condition -> previous dose -> observed/predicted response -> reason for change -> next dose
"""
from __future__ import annotations

import sqlite3
import zlib
from contextlib import closing
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config as C
from .data import load_cohort
from .evaluate import interpret
from .features import patient_features, regimen_from_row, sample_levels
from .models import bayesian_update
from .pk import DosingHistory, Regimen, concentration, steady_state
from .predict import InputError, VancoPredictor

DB_FILE = C.ROOT / "data" / "patient_history.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS courses (
  id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL, start_h REAL NOT NULL,
  dose_mg REAL NOT NULL, interval_h REAL NOT NULL, infusion_h REAL NOT NULL, loading_mg REAL NOT NULL DEFAULT 0,
  weight_kg REAL NOT NULL, scr REAL NOT NULL, icu INTEGER NOT NULL, arc INTEGER NOT NULL, rrt INTEGER NOT NULL,
  source TEXT NOT NULL, note TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS levels (
  id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT NOT NULL, course_id INTEGER, time_h REAL NOT NULL,
  conc REAL NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HistoryStore:
    def __init__(self, path=DB_FILE):
        self.path = path
        with closing(self._db()) as db:
            db.executescript(SCHEMA)

    def _db(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    def courses(self, pid: str) -> list[dict]:
        with closing(self._db()) as db:
            return [dict(r) for r in db.execute("SELECT * FROM courses WHERE patient_id=? ORDER BY start_h", (pid,))]

    def levels(self, pid: str) -> list[dict]:
        with closing(self._db()) as db:
            return [dict(r) for r in db.execute("SELECT * FROM levels WHERE patient_id=? ORDER BY time_h", (pid,))]

    def counts(self) -> dict[str, int]:
        with closing(self._db()) as db:
            return {r[0]: r[1] for r in db.execute("SELECT patient_id, COUNT(*) FROM courses GROUP BY patient_id")}

    def add_course(self, pid: str, c: dict) -> int:
        with closing(self._db()) as db, db:
            cur = db.execute("INSERT INTO courses (patient_id,start_h,dose_mg,interval_h,infusion_h,loading_mg,weight_kg,scr,icu,arc,rrt,source,note,created_at)"
                             " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                             (pid, c["start_h"], c["dose_mg"], c["interval_h"], c["infusion_h"], c["loading_mg"], c["weight_kg"], c["scr"],
                              c["icu"], c["arc"], c["rrt"], c["source"], c.get("note"), _now()))
            return int(cur.lastrowid)

    def add_level(self, pid: str, time_h: float, conc: float, source: str, course_id: int | None) -> int:
        with closing(self._db()) as db, db:
            cur = db.execute("INSERT INTO levels (patient_id,course_id,time_h,conc,source,created_at) VALUES (?,?,?,?,?,?)",
                             (pid, course_id, time_h, conc, source, _now()))
            return int(cur.lastrowid)

    def delete_course(self, pid: str, course_id: int) -> None:
        with closing(self._db()) as db, db:
            db.execute("DELETE FROM levels WHERE patient_id=? AND course_id=?", (pid, course_id))
            db.execute("DELETE FROM courses WHERE patient_id=? AND id=?", (pid, course_id))

    def delete_level(self, pid: str, level_id: int) -> None:
        with closing(self._db()) as db, db:
            db.execute("DELETE FROM levels WHERE patient_id=? AND id=?", (pid, level_id))


# ---------------------------------------------------------------------------
class PatientJourney:
    def __init__(self, predictor: VancoPredictor, store: HistoryStore | None = None):
        self.m = predictor
        self.store = store or HistoryStore()
        self.cohort = load_cohort()
        self.R = self.cohort.regimen_table
        self.P = self.cohort.patients.set_index("patient_id")

    # ------------------------------------------------------------- directory
    def directory(self) -> list[dict]:
        n = self.store.counts()
        P = self.cohort.patients
        return [{"id": r.patient_id, "age": int(r.age), "sex": r.sex, "weight_kg": round(float(r.weight_kg), 1),
                 "crcl": round(float(r.CrCL_ml_min)), "icu": int(r.ICU_flag), "rrt": int(r.RRT_flag), "arc": int(r.ARC_flag),
                 "courses": 1 + n.get(r.patient_id, 0)} for r in P.itertuples()]

    def _patient_row(self, pid: str):
        if pid not in self.P.index:
            raise KeyError(pid)
        return self.P.loc[pid]

    # ------------------------------------------------------------- building blocks
    def _baseline(self, pid: str):
        p = self._patient_row(pid)
        r = self.R[(self.R.patient_id == pid) & (self.R.regimen_type == "standard")].iloc[0]
        g = regimen_from_row(r)
        lv = sample_levels(self.cohort.profile(r.regimen_id), g)
        course = {"id": None, "start_h": 0.0, "dose_mg": float(g.dose_mg), "interval_h": float(g.interval_h),
                  "infusion_h": float(g.infusion_duration_h), "loading_mg": 0.0, "weight_kg": float(p.weight_kg),
                  "scr": float(p.SCr_mg_dl), "icu": int(p.ICU_flag), "arc": int(p.ARC_flag), "rrt": int(p.RRT_flag),
                  "source": "dataset", "note": "Standard weight-based regimen recorded in the dataset."}
        levels = [{"id": None, "course_id": None, "time_h": float(t), "conc": float(c), "source": "dataset"} for t, c in lv]
        return course, levels

    def _raw(self, p, c) -> dict:
        return self.m._patient({"age": p.age, "sex": p.sex, "weight_kg": c["weight_kg"], "height_cm": p.height_cm,
                                "SCr_mg_dl": c["scr"], "ICU_flag": c["icu"], "ARC_flag": c["arc"], "RRT_flag": c["rrt"]})

    @staticmethod
    def _reg(c) -> Regimen:
        return Regimen(c["dose_mg"], c["interval_h"], c["infusion_h"], c.get("loading_mg") or 0.0)

    def _history(self, courses) -> DosingHistory:
        return DosingHistory(tuple((c["start_h"], self._reg(c)) for c in courses))

    def _estimate(self, raw, courses, levels):
        X = patient_features(pd.DataFrame([raw]))
        prior = self.m.pop.prior(X)[0]
        lv = [(l["time_h"], l["conc"]) for l in levels]
        est = bayesian_update(prior, self.m.pop.omega, self._history(courses), lv, self.m.pop.error_model)
        return est, X, prior

    def _p_target(self, daily, est, target):
        return self.m._p_between(np.log(daily) - est.theta[0], est.log_cl_sd, *target)

    # ------------------------------------------------------------- journey
    def journey(self, pid: str, target=(C.AUC_TARGET_LOW, C.AUC_TARGET_HIGH)) -> dict:
        p = self._patient_row(pid)
        base_course, base_levels = self._baseline(pid)
        courses = [base_course] + self.store.courses(pid)
        levels = sorted(base_levels + self.store.levels(pid), key=lambda l: l["time_h"])
        ends = [c["start_h"] for c in courses[1:]] + [np.inf]
        out_courses = []
        prev = None
        for i, (c, end) in enumerate(zip(courses, ends)):
            raw = self._raw(p, c)
            g = self._reg(c)
            before = [l for l in levels if l["time_h"] < c["start_h"]]
            during = [l for l in levels if c["start_h"] <= l["time_h"] < end]
            hist_before = courses[: i + 1]
            est, X, prior = self._estimate(raw, hist_before, before)
            CL, V1, V2, Q = est.params
            auc = g.daily_dose_mg / CL
            ss = steady_state(g, CL, V1, V2, Q)
            rec = self.m.recommend(est, raw, target, current=prev["regimen_obj"] if prev else None)
            ex = self.m.b.explainer.explain(X, raw, g.daily_dose_mg, est.theta[0] - prior[0], len(before))
            for f in ex["factors"]:  # recommended dose is proportional to clearance: opposite of the exposure effect
                f["dose_effect_pct"] = (1 / (1 + f["effect_pct"] / 100) - 1) * 100
            if ex.get("levels_adjustment"):
                e = ex["levels_adjustment"]["effect_pct"]
                ex["levels_adjustment"]["dose_effect_pct"] = (1 / (1 + e / 100) - 1) * 100
            # response observed during this course
            response = None
            if during:
                est_after, _, _ = self._estimate(raw, hist_before, before + during)
                t = np.array([l["time_h"] for l in during])
                expected = concentration(t, self._history(hist_before), *est.params)
                auc_after = g.daily_dose_mg / est_after.params[0]
                ratio = float(np.mean(np.array([l["conc"] for l in during]) / np.maximum(expected, 0.1)))
                response = {"levels": [{**l, "expected": float(e)} for l, e in zip(during, expected)],
                            "auc24": float(auc_after), "interpretation": interpret(auc_after),
                            "clearance_L_h": float(est_after.params[0]),
                            "levels_vs_expected_pct": (ratio - 1) * 100,
                            "auc_change_pct": (auc_after / auc - 1) * 100}
            truth = float(g.daily_dose_mg / p.true_CL_L_h) if "true_CL_L_h" in p else None
            item = {
                "index": i + 1, "id": c["id"], "start_h": c["start_h"], "end_h": None if np.isinf(end) else float(end),
                "source": c["source"], "note": c.get("note"),
                "condition": {"weight_kg": c["weight_kg"], "SCr_mg_dl": c["scr"], "CrCL_ml_min": raw["CrCL_ml_min"],
                              "ICU_flag": c["icu"], "ARC_flag": c["arc"], "RRT_flag": c["rrt"]},
                "regimen": {"dose_mg": g.dose_mg, "interval_h": g.interval_h, "infusion_duration_h": g.infusion_duration_h,
                            "loading_dose_mg": g.loading_dose_mg, "daily_dose_mg": g.daily_dose_mg,
                            "mg_per_kg": g.dose_mg / c["weight_kg"]},
                "prediction": {"auc24": float(auc), "interpretation": interpret(auc), "peak_ss": ss["peak_ss_mg_L"],
                               "trough_ss": ss["trough_ss_mg_L"], "clearance_L_h": float(CL), "log_cl_sd": est.log_cl_sd,
                               "probability_in_target": self._p_target(g.daily_dose_mg, est, target),
                               "basis_levels": len(before)},
                "recommendation_at_decision": rec["best"],
                "response": response,
                "explanation": ex,
                "simulator_true_auc24": truth,
            }
            item["reason"] = self._reason(item, prev, rec)
            item["flags"] = self._flags(item, prev, raw)
            item["regimen_obj"] = g
            out_courses.append(item)
            prev = item
        # Next recommended dose: everything known so far, latest condition
        last = courses[-1]
        raw = self._raw(p, last)
        est, X, prior = self._estimate(raw, courses, levels)
        rec = self.m.recommend(est, raw, target, current=out_courses[-1]["regimen_obj"])
        ex = self.m.b.explainer.explain(X, raw, rec["best"]["daily_dose_mg"], est.theta[0] - prior[0], len(levels))
        for f in ex["factors"]:
            f["dose_effect_pct"] = (1 / (1 + f["effect_pct"] / 100) - 1) * 100
        if ex.get("levels_adjustment"):
            e = ex["levels_adjustment"]["effect_pct"]
            ex["levels_adjustment"]["dose_effect_pct"] = (1 / (1 + e / 100) - 1) * 100
        last_level = max([l["time_h"] for l in levels], default=0.0)
        next_start = float(np.ceil(max(last["start_h"] + 48.0, last_level + 2.0)))
        cur_auc = out_courses[-1]["response"]["auc24"] if out_courses[-1]["response"] else out_courses[-1]["prediction"]["auc24"]
        nxt = {"start_h": next_start, "recommendation": rec, "explanation": ex,
               "clearance_L_h": float(est.params[0]), "log_cl_sd": est.log_cl_sd, "n_levels": len(levels),
               "reason": self._next_reason(out_courses[-1], cur_auc, rec, target),
               "condition": {"weight_kg": last["weight_kg"], "SCr_mg_dl": last["scr"], "ICU_flag": last["icu"],
                             "ARC_flag": last["arc"], "RRT_flag": last["rrt"]}}
        # Concentration timeline with the latest individual estimate, projected 72 h into the next course
        proj = courses + [{"start_h": next_start, "dose_mg": rec["best"]["dose_mg"], "interval_h": rec["best"]["interval_h"],
                           "infusion_h": rec["best"]["infusion_duration_h"], "loading_mg": 0.0}]
        horizon = next_start + 72.0
        grid = np.round(np.arange(0, horizon + 1e-9, 0.5), 2)
        curve = concentration(grid, self._history(proj), *est.params)
        for c in out_courses:
            c.pop("regimen_obj")
        return {
            "patient": {"id": pid, "age": int(p.age), "sex": p.sex, "height_cm": round(float(p.height_cm), 1),
                        "weight_kg": round(float(p.weight_kg), 1), "SCr_mg_dl": round(float(p.SCr_mg_dl), 2),
                        "CrCL_ml_min": round(float(p.CrCL_ml_min)), "ICU_flag": int(p.ICU_flag), "RRT_flag": int(p.RRT_flag),
                        "ARC_flag": int(p.ARC_flag), "MIC_mg_l": float(p.MIC_mg_l), "split": p.split,
                        "subpopulation": p.subpopulation},
            "target": list(target),
            "courses": out_courses,
            "levels": levels,
            "next": nxt,
            "timeline": {"t": grid.tolist(), "c": curve.round(2).tolist(), "projection_start_h": next_start},
        }

    # ------------------------------------------------------------- narrative
    @staticmethod
    def _fmt_reg(r):
        return f"{r['dose_mg']:,.0f} mg every {r['interval_h']:.0f} h"

    def _reason(self, item, prev, rec) -> list[str]:
        pr, rg = item["prediction"], item["regimen"]
        out = []
        if prev is None:
            out.append(f"Starting regimen recorded in the dataset: standard weight-based dosing of {rg['mg_per_kg']:.1f} mg/kg "
                       f"every {rg['interval_h']:.1f} h. It was chosen without kidney function or model input.")
            out.append(f"With the patient's characteristics alone, VancoDose would have predicted AUC24 {pr['auc24']:,.0f} mg·h/L "
                       f"({pr['interpretation']['label'].lower()}).")
            if rec["best"]:
                out.append(f"The covariate-only recommendation would have been {rec['best']['label']}.")
            return out
        basis = prev["response"] or {"auc24": prev["prediction"]["auc24"], "interpretation": prev["prediction"]["interpretation"]}
        n = pr["basis_levels"]
        out.append(f"Course {prev['index']} ({self._fmt_reg(prev['regimen'])}) was estimated at AUC24 {basis['auc24']:,.0f} mg·h/L "
                   f"({basis['interpretation']['label'].lower()}), " + (f"based on {n} measured level{'s' if n != 1 else ''}." if n else "from patient characteristics only."))
        if prev["response"]:
            d = prev["response"]["levels_vs_expected_pct"]
            out.append(f"Measured levels averaged {abs(d):.0f}% {'above' if d > 0 else 'below'} what was expected, so the clearance estimate moved to "
                       f"{prev['response']['clearance_L_h']:.2f} L/h.")
        cond, pc = item["condition"], prev["condition"]
        if abs(cond["SCr_mg_dl"] - pc["SCr_mg_dl"]) >= 0.1:
            out.append(f"Serum creatinine changed from {pc['SCr_mg_dl']:.2f} to {cond['SCr_mg_dl']:.2f} mg/dL "
                       f"(CrCL {pc['CrCL_ml_min']:.0f} → {cond['CrCL_ml_min']:.0f} mL/min).")
        ch = (rg["daily_dose_mg"] / prev["regimen"]["daily_dose_mg"] - 1) * 100
        if abs(ch) >= 1:
            out.append(f"Daily dose {'increased' if ch > 0 else 'reduced'} by {abs(ch):.0f}% to {rg['daily_dose_mg']:,.0f} mg/day, "
                       f"giving a predicted AUC24 of {pr['auc24']:,.0f} ({pr['probability_in_target'] * 100:.0f}% chance within target).")
        else:
            out.append(f"Daily dose kept at {rg['daily_dose_mg']:,.0f} mg/day; predicted AUC24 {pr['auc24']:,.0f}.")
        if item["source"] == "recommended":
            out.append("This regimen is the one VancoDose recommended at the time.")
        elif rec["best"] and abs(rec["best"]["daily_dose_mg"] - rg["daily_dose_mg"]) > 1:
            out.append(f"Clinician-chosen regimen; VancoDose had recommended {rec['best']['label']} (predicted AUC24 {rec['best']['auc24']:,.0f}).")
        if item.get("note"):
            out.append(f"Clinician note: {item['note']}")
        return out

    @staticmethod
    def _next_reason(last, cur_auc, rec, target) -> list[str]:
        b = rec["best"]
        cat = interpret(cur_auc)
        out = [f"Latest estimate for the current regimen: AUC24 {cur_auc:,.0f} mg·h/L ({cat['label'].lower()})."]
        if cat["category"] == "within" and abs(b["daily_dose_mg"] - last["regimen"]["daily_dose_mg"]) / last["regimen"]["daily_dose_mg"] < 0.1:
            out.append("The current regimen is already on target; no change needed. Recheck levels if kidney function changes.")
        else:
            out.append(f"{b['label']} aims for AUC24 ≈ {rec['goal_auc24']:,.0f} within the {target[0]:.0f}–{target[1]:.0f} target "
                       f"(predicted {b['auc24']:,.0f}, {b['probability_in_target'] * 100:.0f}% chance within target).")
        if rec.get("note"):
            out.append(rec["note"])
        return out

    @staticmethod
    def _flags(item, prev, raw) -> list[dict]:
        f = []
        pr = item["prediction"]
        auc = item["response"]["auc24"] if item["response"] else pr["auc24"]
        if auc > C.AUC_TOXICITY:
            f.append({"level": "high", "text": f"AUC24 above {C.AUC_TOXICITY:.0f} mg·h/L: higher risk of kidney injury"})
        elif auc > C.AUC_TARGET_HIGH:
            f.append({"level": "medium", "text": "Exposure above target"})
        elif auc < C.AUC_TARGET_LOW:
            f.append({"level": "medium", "text": "Exposure below target: risk of treatment failure"})
        if pr["trough_ss"] > 20:
            f.append({"level": "medium", "text": f"Predicted steady-state trough {pr['trough_ss']:.0f} mg/L (> 20)"})
        if prev:
            a, b = prev["condition"]["SCr_mg_dl"], item["condition"]["SCr_mg_dl"]
            if b - a >= 0.3 or b >= 1.5 * a:
                f.append({"level": "high", "text": f"Serum creatinine rose {a:.2f} → {b:.2f} mg/dL: possible acute kidney injury"})
        if raw["RRT_flag"]:
            f.append({"level": "info", "text": "On renal replacement therapy: few such patients in training data"})
        if raw["SCr_mg_dl"] > C.MAX_SCR_MG_DL:
            f.append({"level": "high", "text": f"Serum creatinine above {C.MAX_SCR_MG_DL} mg/dL is outside the model's scope"})
        return f

    # ------------------------------------------------------------- mutations
    def add_course(self, pid: str, body: dict) -> int:
        p = self._patient_row(pid)
        j_courses = [self._baseline(pid)[0]] + self.store.courses(pid)
        last = j_courses[-1]
        try:
            start = float(body["start_h"])
            reg = body["regimen"]
            g = Regimen(float(reg["dose_mg"]), float(reg["interval_h"]),
                        float(reg.get("infusion_duration_h") or max(1.0, float(reg["dose_mg"]) / 1000)), float(reg.get("loading_dose_mg") or 0))
            g.validate()
        except (KeyError, TypeError, ValueError) as e:
            raise InputError(f"Check the regimen: {e}") from None
        if not last["start_h"] + 6 <= start <= last["start_h"] + 24 * 14:
            raise InputError(f"The new course must start 6 h–14 days after the previous one (after {last['start_h'] + 6:.0f} h).")
        cl = {**{k: last[k] for k in ("weight_kg", "scr", "icu", "arc", "rrt")}, **{k: v for k, v in (body.get("condition") or {}).items() if v is not None}}
        raw = self.m._patient({"age": p.age, "sex": p.sex, "weight_kg": cl["weight_kg"], "SCr_mg_dl": cl["scr"],
                               "ICU_flag": cl["icu"], "ARC_flag": cl["arc"], "RRT_flag": cl["rrt"]})  # validates ranges
        course = {"start_h": start, "dose_mg": g.dose_mg, "interval_h": g.interval_h, "infusion_h": g.infusion_duration_h,
                  "loading_mg": g.loading_dose_mg, "weight_kg": raw["weight_kg"], "scr": raw["SCr_mg_dl"], "icu": raw["ICU_flag"],
                  "arc": raw["ARC_flag"], "rrt": raw["RRT_flag"], "source": "recommended" if body.get("accepted") else "clinician",
                  "note": (body.get("note") or "").strip()[:500] or None}
        cid = self.store.add_course(pid, course)
        if body.get("simulate_levels"):
            self._simulate_levels(pid, p, j_courses + [course], cid)
        return cid

    def _simulate_levels(self, pid, p, courses, cid):
        """Synthetic patients only: draw a peak and trough around dose 3-4 of the new course from the
        simulator's true parameters and assay-noise model (clearly labelled as simulated)."""
        c = courses[-1]
        g = self._reg(c)
        hist = self._history(courses)
        # Clearance follows kidney function: scale the latent CL by the change in CrCL (renal fraction assumed ~1)
        base_crcl = float(p.CrCL_ml_min)
        crcl = (140 - p.age) * c["weight_kg"] / (72 * c["scr"]) * (0.85 if p.sex == "F" else 1.0)
        cl = float(p.true_CL_L_h) * (crcl / base_crcl) ** 0.8
        params = (cl, float(p.true_Vd1_L), float(p.true_Vd2_L), float(p.true_Q_L_h))
        times = [c["start_h"] + 2 * g.interval_h + g.infusion_duration_h + 1.0, c["start_h"] + 3 * g.interval_h - 0.1]
        true = concentration(times, hist, *params)
        rng = np.random.default_rng(zlib.crc32(f"{pid}-{c['start_h']}".encode()))
        add, prop = self.m.pop.error_model
        for t, y in zip(times, true):
            obs = max(0.0, y + rng.normal(0, np.sqrt(add ** 2 + (prop * y) ** 2)))
            self.store.add_level(pid, round(float(t), 2), round(float(obs), 1), "simulated", cid)

    def add_level(self, pid: str, time_h: float, conc: float) -> int:
        self._patient_row(pid)
        courses = [self._baseline(pid)[0]] + self.store.courses(pid)
        if not (0 < time_h <= courses[-1]["start_h"] + 24 * 14) or not (0 <= conc < 200):
            raise InputError("Enter a time after the first dose and a concentration between 0 and 200 mg/L.")
        cid = next((c["id"] for c in reversed(courses) if c["start_h"] <= time_h), None)
        return self.store.add_level(pid, time_h, conc, "measured", cid)

    def delete_course(self, pid: str, course_id: int) -> None:
        stored = self.store.courses(pid)
        if not stored or stored[-1]["id"] != course_id:
            raise InputError("Only the most recent added course can be removed, to keep the timeline consistent.")
        self.store.delete_course(pid, course_id)
