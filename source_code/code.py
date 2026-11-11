"""
Early-Warning Fall Prediction (UMAFall) - complete implementation (v2, incl. strict metric, persistence and live demo) as a plain script.
Run from the folder that contains ./data :  python fall_prediction_prml.py
Env vars: UMAFALL_DIR, OUT_DIR=results, FAST=1 (quick run), REBUILD=1 (ignore cache)
"""
import os; os.environ.setdefault("MPL_BACKEND", "Agg")

import os, re, glob, time, warnings, pickle
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
if os.environ.get("MPL_BACKEND"): matplotlib.use(os.environ["MPL_BACKEND"])
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})

# ----------------------------- CONFIG (edit here) -----------------------------
def find_data_dir():
    # Use UMAFALL_DIR if set, else look inside ./data (any depth) for the folder that holds the UMAFall_*.csv files.
    for base in [os.environ.get("UMAFALL_DIR"), "data/UMAFall", "data", "."]:
        if base and Path(base).exists():
            hit = next(Path(base).rglob("UMAFall_Subject_*.csv"), None)
            if hit is not None: return hit.parent
    return Path("data/UMAFall")
DATA_DIR = find_data_dir()
OUT_DIR  = Path(os.environ.get("OUT_DIR", "results")); OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE    = OUT_DIR / "trials_cache.pkl"
FAST     = os.environ.get("FAST", "0") == "1"      # quick dry-run mode

FS      = 20.0          # common sampling rate (Hz) - SensorTags run at ~20 Hz
WIN     = 20            # decision window = 1.0 s
CTX     = 40            # context used for 'repetitiveness' features = 2.0 s
POS_WIDTH = 0.30        # positives = windows ending within 0.3 s around (t_impact - tau)
AMBIG_BEFORE = 2.0      # fall-trial windows ending 2 s+ before impact are 'normal' negatives; the gap is dropped as ambiguous
LEADS   = [0.0, 0.3, 0.6] if FAST else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
MAIN_POS = "CHEST"      # main sensor position (see Section 9 for all positions)
C_FN, C_FP = 10.0, 1.0  # a missed fall is 10x worse than a false alarm (swept in Section 8)
N_SPLITS, N_INNER = (3, 2) if FAST else (5, 3)
SEED = 42
POS = {0: "RIGHTPOCKET", 1: "CHEST", 2: "WAIST", 3: "WRIST", 4: "ANKLE"}
STYPE = {0: "acc", 1: "gyr", 2: "mag"}
print("DATA_DIR :", DATA_DIR.resolve(), "| CSV files:", len(list(DATA_DIR.glob("UMAFall_Subject_*.csv"))))
print("FAST mode:", FAST, "| lead times (s):", LEADS)

NAME_RE = re.compile(r"UMAFall_Subject_(\d+)_(ADL|Fall)_(.+?)_(\d+)_(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)\.csv$")
COLS = ["t", "n", "x", "y", "z", "type", "sid"]

def load_trial(path):
    m = NAME_RE.match(Path(path).name)
    if m is None: return None
    subj, kind, act, trial, stamp = m.groups()
    df = pd.read_csv(path, sep=";", comment="%", header=None, usecols=range(7), names=COLS, on_bad_lines="skip")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    streams = {}
    for (sid, typ), g in df.groupby(["sid", "type"]):
        sid, typ = int(sid), int(typ)
        if typ == 2 or sid not in POS or len(g) < 2 * WIN: continue       # skip magnetometer / junk
        g = g.sort_values("n", kind="mergesort")
        idx = np.arange(len(g)); ts = g["t"].values / 1000.0
        slope, icpt = np.polyfit(idx, ts, 1)                               # clean, evenly spaced time axis
        if slope <= 0: continue
        t = icpt + slope * idx; xyz = g[["x", "y", "z"]].values.astype(float)
        fs_est = 1.0 / slope
        if fs_est > 2 * FS:                                                # phone @200 Hz -> anti-alias then decimate
            xyz = uniform_filter1d(xyz, size=int(round(fs_est / FS)), axis=0, mode="nearest")
        streams[(POS[sid], STYPE[typ])] = (t, xyz)
    if not streams: return None
    t0 = min(s[0][0] for s in streams.values())
    T = float(np.median([s[0][-1] - t0 for s in streams.values()]))
    grid = np.arange(0, T, 1 / FS)
    acc, gyr = {}, {}
    for (pos, kindS), (t, xyz) in streams.items():
        arr = np.column_stack([np.interp(grid, t - t0, xyz[:, k]) for k in range(3)])
        (acc if kindS == "acc" else gyr)[pos] = arr
    rec = dict(name=Path(path).name, subject=int(subj), is_fall=int(kind == "Fall"), activity=act, trial=int(trial),
               N=len(grid), acc=acc, gyr=gyr, t_imp=np.nan, ref_pos=None)
    if rec["is_fall"]:                                                     # impact = peak SMV of the best available reference sensor
        for ref in ["CHEST", "WAIST", "RIGHTPOCKET", "WRIST", "ANKLE"]:
            if ref in acc:
                smv = np.linalg.norm(acc[ref], axis=1)
                rec["t_imp"], rec["ref_pos"], rec["peak_g"] = grid[int(np.argmax(smv))], ref, float(smv.max()); break
    return rec

files = sorted(glob.glob(str(DATA_DIR / "UMAFall_*.csv")))
print("CSV files found:", len(files))
assert len(files) > 0, "No CSV files found - check DATA_DIR (see run instructions PDF)."
if CACHE.exists() and os.environ.get("REBUILD") != "1":
    trials = pickle.load(open(CACHE, "rb")); print("loaded cache:", len(trials), "trials")
else:
    t0 = time.time()
    trials = [r for r in Parallel(n_jobs=-1)(delayed(load_trial)(f) for f in files) if r is not None]
    pickle.dump(trials, open(CACHE, "wb")); print(f"parsed {len(trials)} trials in {time.time()-t0:.0f}s (cached to {CACHE})")

meta = pd.DataFrame([{k: r[k] for k in ["name", "subject", "is_fall", "activity", "trial", "N", "t_imp"]} |
                     {"peak_g": r.get("peak_g", np.nan), **{f"has_{p}": (p in r["acc"]) for p in POS.values()}} for r in trials])
print("trials:", len(meta), "| falls:", int(meta.is_fall.sum()), "| ADL:", int((1-meta.is_fall).sum()), "| subjects:", meta.subject.nunique())
print("\nTrials per activity:\n", meta.groupby(["is_fall", "activity"]).size().to_string())
print("\nSensor availability (number of trials having that accelerometer):\n", meta[[c for c in meta if c.startswith('has_')]].sum().to_string())
print("\nTrial duration (s): mean %.1f, min %.1f, max %.1f" % (meta.N.mean()/FS, meta.N.min()/FS, meta.N.max()/FS))
print("\nFall impact time (s from start): mean %.2f, min %.2f, max %.2f" % (meta.t_imp.mean(), meta.t_imp.min(), meta.t_imp.max()))
print("Fall peak SMV (g): median %.2f | trials with peak < 1.5 g (suspicious): %d" % (meta.peak_g.median(), (meta.peak_g < 1.5).sum()))
print("\nFalls per subject:\n", meta[meta.is_fall == 1].groupby("subject").size().to_string())

def smv_of(r, pos=MAIN_POS):
    return np.linalg.norm(r["acc"][pos], axis=1) if pos in r["acc"] else None
def first(pred):
    return next(r for r in trials if pred(r) and MAIN_POS in r["acc"])
ex = [("Backward fall", first(lambda r: r["is_fall"] and "backward" in r["activity"])),
      ("Forward fall",  first(lambda r: r["is_fall"] and "forward" in r["activity"])),
      ("Jogging (ADL)", first(lambda r: r["activity"] == "Jogging")),
      ("Walking (ADL)", first(lambda r: r["activity"] == "Walking"))]
fig, axs = plt.subplots(len(ex), 1, figsize=(9, 8), sharex=True, sharey=True)
for ax, (title, r) in zip(axs, ex):
    s = smv_of(r); ax.plot(np.arange(len(s)) / FS, s, lw=1.1, color="#b2182b" if r["is_fall"] else "#2166ac")
    ax.axhline(1, color="grey", ls="--", lw=.8)
    if r["is_fall"]: ax.axvline(r["t_imp"], color="k", ls=":", label="impact (peak SMV)"); ax.legend(loc="upper right")
    ax.set_ylabel("SMV (g)"); ax.set_title(f"{title} - chest accelerometer  [{r['name'][:38]}]", fontsize=9)
axs[-1].set_xlabel("time (s)"); plt.tight_layout(); plt.savefig(OUT_DIR / "fig01_signals.png", dpi=150); plt.show()

def extract(A, G=None):
    # A: (n, CTX, 3) accel windows (g); G: (n, CTX, 3) gyro windows or None. Returns dict of feature arrays.
    smv = np.linalg.norm(A, axis=2); w = smv[:, -WIN:]; d = np.diff(w, axis=1) * FS
    prev = smv[:, :-WIN]
    f = {}
    f["smv_mean"] = w.mean(1); f["smv_std"] = w.std(1); f["smv_min"] = w.min(1); f["smv_max"] = w.max(1)
    f["smv_range"] = f["smv_max"] - f["smv_min"]; f["smv_last"] = w[:, -1]
    f["smv_min_last05"] = w[:, -10:].min(1); f["smv_std_last05"] = w[:, -10:].std(1)
    f["smv_argmin_frac"] = w.argmin(1) / (WIN - 1)
    f["frac_below_07"] = (w < 0.7).mean(1); f["frac_above_15"] = (w > 1.5).mean(1)
    f["slope_last"] = (w[:, -1] - w[:, -6]) * FS / 5
    f["jerk_max"] = np.abs(d).max(1); f["jerk_mean"] = np.abs(d).mean(1)
    f["prev_mean"] = prev.mean(1); f["prev_std"] = prev.std(1); f["dip_depth"] = f["prev_mean"] - f["smv_min"]
    a1 = A[:, -WIN:-WIN + 5].mean(1); a2 = A[:, -5:].mean(1)
    cosang = (a1 * a2).sum(1) / (np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1) + 1e-9)
    f["tilt_change_deg"] = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    f["ctx_std"] = smv.std(1); f["ctx_max"] = smv.max(1); f["ctx_mad"] = np.abs(smv - smv.mean(1, keepdims=True)).mean(1)
    mid = smv[:, 1:-1]; f["ctx_peaks"] = ((mid > smv[:, :-2]) & (mid >= smv[:, 2:]) & (mid > 1.5)).sum(1)
    if G is not None:
        g = np.linalg.norm(G, axis=2); gw = g[:, -WIN:]
        f["gyr_mean"] = gw.mean(1); f["gyr_max"] = gw.max(1); f["gyr_std"] = gw.std(1); f["gyr_last"] = gw[:, -1]
        f["gyr_slope"] = (gw[:, -1] - gw[:, -6]) * FS / 5; f["gyr_ctx_max"] = g.max(1)
    return f

def build_window_table(trials, pos, use_gyro=True):
    # One row per causal window. Returns (meta DataFrame, feature DataFrame).
    M, F = [], []
    for ti, r in enumerate(trials):
        if pos not in r["acc"] or (use_gyro and pos not in r["gyr"]): continue
        acc = r["acc"][pos]; N = len(acc)
        pad = lambda x: np.vstack([np.repeat(x[:1], CTX - 1, 0), x])
        stride = 1 if r["is_fall"] else 2
        ends = np.arange(WIN - 1, N, stride)
        A = sliding_window_view(pad(acc), CTX, axis=0).transpose(0, 2, 1)[ends]
        G = sliding_window_view(pad(r["gyr"][pos]), CTX, axis=0).transpose(0, 2, 1)[ends] if use_gyro else None
        F.append(pd.DataFrame(extract(A, G)))
        M.append(pd.DataFrame({"trial": ti, "subject": r["subject"], "is_fall": r["is_fall"], "activity": r["activity"],
                               "t_end": ends / FS, "t_imp": r["t_imp"], "dur": N / FS}))
    return pd.concat(M, ignore_index=True), pd.concat(F, ignore_index=True).astype(np.float32)

t0 = time.time()
Wm, Wf = build_window_table(trials, MAIN_POS, use_gyro=True)
FEATS = list(Wf.columns)
print(f"{MAIN_POS}: {len(Wm):,} windows x {len(FEATS)} features from {Wm.trial.nunique()} trials ({time.time()-t0:.0f}s)")
Wf.describe().T[["mean", "std", "min", "max"]].round(3).head(12)

def assign_labels(M, tau, pos_width=POS_WIDTH, ctx=AMBIG_BEFORE):
    lab = np.full(len(M), np.nan)
    f = M.is_fall.values == 1; te = M.t_end.values; ti = M.t_imp.values
    lab[~f] = 0
    lab[f & (te < ti - ctx)] = 0
    lab[f & (te <= ti - tau + 1e-9) & (te >= ti - tau - pos_width - 1e-9)] = 1
    return lab

rows = []
for tau in LEADS:
    l = assign_labels(Wm, tau); rows.append({"tau (s)": tau, "positives": int((l == 1).sum()), "negatives": int((l == 0).sum()),
                                           "ignored": int(np.isnan(l).sum()), "pos : neg": f"1 : {(l==0).sum()/max((l==1).sum(),1):.0f}"})
print(pd.DataFrame(rows).to_string(index=False))

rows = []
for tau in [0.0, 0.3]:
    lab = assign_labels(Wm, tau); m = ~np.isnan(lab)
    for fe, sign, nm in [("smv_max", 1, "peak SMV (max)"), ("smv_min", -1, "free-fall dip (-min SMV)"), ("jerk_max", 1, "max jerk"),
                         ("tilt_change_deg", 1, "tilt change"), ("gyr_max", 1, "max gyro"), ("ctx_peaks", -1, "-(#peaks in 2 s)")]:
        rows.append({"tau": tau, "rule": nm, "ROC-AUC": roc_auc_score(lab[m], sign * Wf[fe].values[m]),
                     "PR-AUC": average_precision_score(lab[m], sign * Wf[fe].values[m])})
base = pd.DataFrame(rows); print(base.pivot(index="rule", columns="tau", values=["ROC-AUC", "PR-AUC"]).round(3))

fig, ax = plt.subplots(figsize=(10, 3.6))
tmp = Wm.assign(smv_max=Wf.smv_max.values)
tmp = tmp[(tmp.is_fall == 0) | ((tmp.t_end >= tmp.t_imp - 0.05) & (tmp.t_end <= tmp.t_imp + 0.05))]   # ADL windows vs fall-at-impact windows
tmp["group"] = np.where(tmp.is_fall == 1, "FALL (at impact)", tmp.activity)
order = tmp.groupby("group").smv_max.median().sort_values().index
ax.boxplot([tmp[tmp.group == g].smv_max for g in order], showfliers=False); ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels(order)
ax.set_ylabel("max SMV in 1 s window (g)"); ax.set_title("Peak magnitude overlaps heavily: vigorous ADLs vs falls")
plt.xticks(rotation=40, ha="right"); plt.tight_layout(); plt.savefig(OUT_DIR / "fig02_peak_overlap.png", dpi=150); plt.show()

def make_rf():  return RandomForestClassifier(n_estimators=60 if FAST else 150, min_samples_leaf=5, max_samples=0.3,
                                              class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
def make_lr():  return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))
def make_hgb(): return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=150, class_weight="balanced", random_state=SEED)

def best_threshold(y, p, c_fn=C_FN, c_fp=C_FP):
    # threshold minimising  c_fn*FN + c_fp*FP  (flagging windows with p >= thr). Includes the 'never alarm' option.
    o = np.argsort(-p, kind="mergesort"); ys = y[o]; ps = p[o]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); P = ys.sum()
    cost = c_fn * (P - tp) + c_fp * fp
    cost = np.where(np.r_[ps[:-1] > ps[1:], True], cost, np.inf)            # only cut where the score changes
    i = int(np.argmin(cost))
    if c_fn * P <= cost[i]: return float(ps[0] + 1e-6)
    return float((ps[i] + (ps[i + 1] if i + 1 < len(ps) else 0.0)) / 2)

def run_cv(Mx, X, tau, make_model, c_fn=C_FN, c_fp=C_FP, n_splits=N_SPLITS, n_inner=N_INNER, nested_threshold=True):
    # Subject-level CV. Returns out-of-fold probability for EVERY window of held-out subjects + the nested threshold used.
    lab = assign_labels(Mx, tau); idx = np.where(~np.isnan(lab))[0]; y = lab.astype(float)
    groups = Mx.subject.values; X = np.asarray(X)
    oof = np.full(len(Mx), np.nan); thr = np.full(len(Mx), np.nan)
    cv = StratifiedGroupKFold(n_splits, shuffle=True, random_state=SEED)
    for tr_i, te_i in cv.split(idx, y[idx], groups[idx]):
        tr = idx[tr_i]; model = make_model().fit(X[tr], y[tr])
        t = 0.5
        if nested_threshold:
            p_in = np.zeros(len(tr)); inner = StratifiedGroupKFold(n_inner, shuffle=True, random_state=SEED)
            for a, b in inner.split(tr, y[tr], groups[tr]):
                p_in[b] = make_model().fit(X[tr[a]], y[tr[a]]).predict_proba(X[tr[b]])[:, 1]
            t = best_threshold(y[tr], p_in, c_fn, c_fp)
        te_all = np.where(np.isin(groups, np.unique(groups[idx[te_i]])))[0]  # ALL windows of the held-out subjects
        oof[te_all] = model.predict_proba(X[te_all])[:, 1]; thr[te_all] = t
    return lab, oof, thr

# ---- trial-level evaluation helpers -------------------------------------------------------------
def trial_index(Mx):
    return {t: np.where(Mx.trial.values == t)[0] for t in Mx.trial.unique()}

def trial_metrics(Mx, tidx, oof, thr, tau, refractory=1.0):
    # Fall caught early: any alarm with window-end <= t_imp - tau. ADL: alarm events (merged within 1 s) per minute.
    caught, leads, fall_rows = [], [], []
    ev, minutes, adl_alarm_trials, n_adl = 0, 0.0, 0, 0
    thr = np.broadcast_to(thr, oof.shape)
    for t, ix in tidx.items():
        if np.isnan(oof[ix]).any(): continue
        te = Mx.t_end.values[ix]; flag = oof[ix] >= thr[ix]
        if Mx.is_fall.values[ix[0]] == 1:
            ti = Mx.t_imp.values[ix[0]]; ok = flag & (te <= ti - tau + 1e-9)
            caught.append(bool(ok.any())); leads.append(ti - te[ok].min() if ok.any() else np.nan)
            fall_rows.append((t, Mx.activity.values[ix[0]], bool(ok.any())))
        else:
            tt = te[flag]; n = 0; last = -9e9
            for x in tt:
                if x - last > refractory: n += 1
                last = x
            ev += n; minutes += Mx.dur.values[ix[0]] / 60; adl_alarm_trials += int(n > 0); n_adl += 1
    leads = np.array(leads, float)
    return {"fall_recall_trial": float(np.mean(caught)) if caught else np.nan,
            "median_lead_s": float(np.nanmedian(leads)) if np.isfinite(leads).any() else np.nan,
            "adl_alarms_per_min": ev / max(minutes, 1e-9), "adl_trials_with_alarm": adl_alarm_trials / max(n_adl, 1),
            "n_falls": len(caught), "n_adl": n_adl}, fall_rows, leads

def window_metrics(lab, oof, thr, c_fn=C_FN, c_fp=C_FP):
    m = ~np.isnan(lab) & ~np.isnan(oof); y = lab[m].astype(int); p = oof[m]; pred = p >= thr[m]
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "recall": tp / max(tp + fn, 1),
            "precision": tp / max(tp + fp, 1), "FP": fp, "FN": fn, "cost": c_fn * fn + c_fp * fp}

tidx = trial_index(Wm); RES, STORE = [], {}
t0 = time.time()
for tau in LEADS:
    lab, oof, thr = run_cv(Wm, Wf.values, tau, make_rf)
    wm = window_metrics(lab, oof, thr); tm, fall_rows, leads = trial_metrics(Wm, tidx, oof, thr, tau)
    RES.append({"tau": tau, **wm, **tm}); STORE[tau] = dict(lab=lab, oof=oof, thr=thr, fall_rows=fall_rows, leads=leads)
    print(f"tau={tau:.1f}s  ROC-AUC={wm['roc_auc']:.3f}  PR-AUC={wm['pr_auc']:.3f}  | trial recall={tm['fall_recall_trial']:.2f} "
          f"median lead={tm['median_lead_s']:.2f}s  ADL false alarms/min={tm['adl_alarms_per_min']:.2f}   [{time.time()-t0:.0f}s]")
res = pd.DataFrame(RES); res.to_csv(OUT_DIR / "table_main_results.csv", index=False)
res.round(3)

fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
axs[0].plot(res.tau, res.roc_auc, "o-", label="ROC-AUC"); axs[0].plot(res.tau, res.pr_auc, "s-", label="PR-AUC")
axs[0].set_xlabel("required lead time tau (s)"); axs[0].set_ylabel("score"); axs[0].set_title("Ranking quality vs lead time"); axs[0].legend()
axs[1].plot(res.tau, res.fall_recall_trial, "o-", color="#b2182b"); axs[1].set_ylim(0, 1.02)
axs[1].set_xlabel("required lead time tau (s)"); axs[1].set_ylabel("falls caught >= tau before impact"); axs[1].set_title(f"Trial-level fall recall (C_FN:C_FP = {C_FN:.0f}:{C_FP:.0f})")
axs[2].plot(res.tau, res.adl_alarms_per_min, "o-", color="#2166ac"); axs[2].set_xlabel("required lead time tau (s)")
axs[2].set_ylabel("false alarms per minute of ADL"); axs[2].set_title("False-alarm burden needed to achieve that")
plt.tight_layout(); plt.savefig(OUT_DIR / "fig03_tau_tradeoff.png", dpi=150); plt.show()

fig, ax = plt.subplots(figsize=(7.2, 4.8))
sel = [t for t in [0.0, 0.2, 0.4, 0.6, 1.0] if t in STORE] or LEADS
for tau in sel:
    oof = STORE[tau]["oof"]; valid = oof[~np.isnan(oof)]
    thrs = np.unique(np.quantile(valid, np.r_[np.linspace(0.5, 0.99, 30), np.linspace(0.99, 0.9995, 25)]))
    pts = []
    for th in thrs:
        tm, _, _ = trial_metrics(Wm, tidx, oof, th, tau); pts.append((tm["adl_alarms_per_min"], tm["fall_recall_trial"]))
    pts = np.array(pts); ax.plot(pts[:, 0], pts[:, 1], "-o", ms=3, label=f"tau = {tau:.1f} s")
ax.set_xlabel("ADL false alarms per minute"); ax.set_ylabel("fraction of falls caught >= tau before impact"); ax.set_xscale("symlog", linthresh=0.1)
ax.set_title("Operating curves: more lead time -> worse recall at the same false-alarm rate"); ax.legend()
plt.tight_layout(); plt.savefig(OUT_DIR / "fig04_operating_curves.png", dpi=150); plt.show()

TAU_REF = 0.3 if 0.3 in LEADS else LEADS[len(LEADS) // 2]
rows = []
for nm, mk in [("Logistic Regression", make_lr), ("Random Forest", make_rf), ("Hist. Gradient Boosting", make_hgb)]:
    lab, oof, thr = run_cv(Wm, Wf.values, TAU_REF, mk)
    wm = window_metrics(lab, oof, thr); tm, _, _ = trial_metrics(Wm, tidx, oof, thr, TAU_REF)
    rows.append({"model": nm, "ROC-AUC": wm["roc_auc"], "PR-AUC": wm["pr_auc"], "win recall": wm["recall"], "win precision": wm["precision"],
                 "expected cost": wm["cost"], "trial recall": tm["fall_recall_trial"], "ADL alarms/min": tm["adl_alarms_per_min"]})
    print(nm, "done")
cmp_df = pd.DataFrame(rows); cmp_df.to_csv(OUT_DIR / "table_model_comparison.csv", index=False)
print(f"\nTau = {TAU_REF} s"); cmp_df.round(3)

lab = assign_labels(Wm, TAU_REF); m = ~np.isnan(lab)
rf = make_rf().fit(Wf.values[m], lab[m])
imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 6)); imp.tail(18).plot.barh(ax=ax, color="#4575b4")
ax.set_title(f"Random Forest feature importance (tau = {TAU_REF} s)"); plt.tight_layout(); plt.savefig(OUT_DIR / "fig05_feature_importance.png", dpi=150); plt.show()
imp.sort_values(ascending=False).head(8).round(3)

RATIOS = [1, 2, 5, 10, 20, 50, 100]; TAUS_C = [t for t in [0.0, 0.3, 0.6] if t in STORE]; rows = []
for tau in TAUS_C:
    S = STORE[tau]; lab, oof = S["lab"], S["oof"]; m = ~np.isnan(lab) & ~np.isnan(oof)
    for r_ in RATIOS:
        th = best_threshold(lab[m].astype(float), oof[m], c_fn=r_, c_fp=1.0)
        tm, _, _ = trial_metrics(Wm, tidx, oof, th, tau)
        rows.append({"tau": tau, "C_FN:C_FP": r_, "threshold": th, "trial recall": tm["fall_recall_trial"], "ADL alarms/min": tm["adl_alarms_per_min"], "ADL trials w/ alarm": tm["adl_trials_with_alarm"]})
cost_df = pd.DataFrame(rows); cost_df.to_csv(OUT_DIR / "table_cost_sweep.csv", index=False)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for tau in TAUS_C:
    d = cost_df[cost_df.tau == tau]
    axs[0].plot(d["C_FN:C_FP"], d["trial recall"], "o-", label=f"tau={tau:.1f}s"); axs[1].plot(d["C_FN:C_FP"], d["ADL alarms/min"], "o-", label=f"tau={tau:.1f}s")
for a, yl in zip(axs, ["falls caught (trial-level)", "ADL false alarms / min"]): a.set_xscale("log"); a.set_xlabel("cost ratio  C_FN : C_FP"); a.set_ylabel(yl); a.legend()
axs[0].set_title("Higher miss-cost -> catch more falls"); axs[1].set_title("...at the price of more false alarms")
plt.tight_layout(); plt.savefig(OUT_DIR / "fig06_cost_sweep.png", dpi=150); plt.show()
cost_df.round(3)

rows = []
for pos in POS.values():
    Mp, Fp = build_window_table(trials, pos, use_gyro=False)
    if len(Mp) == 0 or Mp.is_fall.sum() == 0: print("skip", pos); continue
    tix = trial_index(Mp); lab, oof, thr = run_cv(Mp, Fp.values, TAU_REF, make_rf)
    wm = window_metrics(lab, oof, thr); tm, _, _ = trial_metrics(Mp, tix, oof, thr, TAU_REF)
    rows.append({"position": pos, "trials used": Mp.trial.nunique(), "ROC-AUC": wm["roc_auc"], "PR-AUC": wm["pr_auc"],
                 "trial recall": tm["fall_recall_trial"], "ADL alarms/min": tm["adl_alarms_per_min"]}); print(pos, "done")
pos_df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False); pos_df.to_csv(OUT_DIR / "table_positions.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 3.6)); ax.bar(pos_df.position, pos_df["PR-AUC"], color="#4575b4"); ax.set_ylabel("PR-AUC"); ax.set_title(f"Sensor position comparison (tau = {TAU_REF} s, accel-only)")
plt.tight_layout(); plt.savefig(OUT_DIR / "fig07_positions.png", dpi=150); plt.show()
pos_df.round(3)

S = STORE[TAU_REF]; oof, thr = S["oof"], S["thr"]
fr = pd.DataFrame(S["fall_rows"], columns=["trial", "activity", "caught"])
fr["fall_type"] = fr.activity.str.replace("Fall", "", regex=False)
print(f"Fall recall by fall type (tau = {TAU_REF} s):\n", fr.groupby("fall_type").caught.agg(["mean", "count"]).round(2).rename(columns={"mean": "recall", "count": "n_trials"}).to_string())

adl_rows = []
for t, ix in tidx.items():
    if Wm.is_fall.values[ix[0]] == 1 or np.isnan(oof[ix]).any(): continue
    adl_rows.append((Wm.activity.values[ix[0]], bool((oof[ix] >= thr[ix]).any())))
adl_err = pd.DataFrame(adl_rows, columns=["activity", "false_alarm"]).groupby("activity").false_alarm.agg(["mean", "count"]).sort_values("mean", ascending=False)
print("\nFraction of ADL trials that raised >=1 false alarm, by activity:\n", adl_err.round(2).rename(columns={"mean": "frac_trials_with_alarm", "count": "n_trials"}).to_string())

leads = S["leads"][np.isfinite(S["leads"])]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
axs[0].hist(leads, bins=15, color="#b2182b"); axs[0].set_xlabel("warning time before impact (s)"); axs[0].set_ylabel("# fall trials"); axs[0].set_title(f"Lead time of first alarm (tau = {TAU_REF} s)")
adl_err["mean"].plot.barh(ax=axs[1], color="#2166ac"); axs[1].set_xlabel("fraction of trials with a false alarm"); axs[1].set_title("False alarms by ADL activity")
plt.tight_layout(); plt.savefig(OUT_DIR / "fig08_error_analysis.png", dpi=150); plt.show()

final = res[["tau", "roc_auc", "pr_auc", "fall_recall_trial", "median_lead_s", "adl_alarms_per_min", "adl_trials_with_alarm"]].round(3)
final.to_csv(OUT_DIR / "table_final_summary.csv", index=False); print(final.to_string(index=False))
r0, rx = res.iloc[0], res[res.tau == TAU_REF].iloc[0]
print(f"\nAuto-summary: as required lead time grows from 0 s to {TAU_REF} s, PR-AUC changes {r0.pr_auc:.2f} -> {rx.pr_auc:.2f} and trial-level fall recall "
      f"{r0.fall_recall_trial:.2f} -> {rx.fall_recall_trial:.2f} (ADL false alarms/min {r0.adl_alarms_per_min:.2f} -> {rx.adl_alarms_per_min:.2f}).")
print("All tables/figures saved in:", OUT_DIR.resolve())

def get_fold_ids(Mx, tau):
    # re-create the exact subject-level folds used in run_cv (deterministic: same labels, groups, seed)
    lab = assign_labels(Mx, tau); idx = np.where(~np.isnan(lab))[0]; y = lab.astype(float); g = Mx.subject.values
    fold = np.full(len(Mx), -1)
    for k, (tr_i, te_i) in enumerate(StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED).split(idx, y[idx], g[idx])):
        fold[np.isin(g, np.unique(g[idx[te_i]]))] = k
    return fold

def smooth_causal(te, s, delta):
    # mean of the scores whose window-end time lies in (t - delta, t]  -> uses only the past
    if delta <= 0: return s
    cs = np.r_[0.0, np.cumsum(s)]; j = np.searchsorted(te, te - delta, side="right"); i = np.arange(len(s))
    return (cs[i + 1] - cs[j]) / (i + 1 - j)

def count_events(times, refractory=1.0):
    n, last = 0, -9e9
    for x in times:
        if x - last > refractory: n += 1
        last = x
    return n

def strict_metrics(Mx, tidx, s_all, thr_row, tau, zone=AMBIG_BEFORE):
    caught, leads, ev, minutes, adl_alarm, n_adl = [], [], 0, 0.0, 0, 0
    for t, ix in tidx.items():
        if np.isnan(s_all[ix]).any(): continue
        te = Mx.t_end.values[ix]; flag = s_all[ix] >= thr_row[ix]
        if Mx.is_fall.values[ix[0]] == 1:
            ti = Mx.t_imp.values[ix[0]]
            ok = flag & (te >= ti - zone) & (te <= ti - tau + 1e-9)                 # actionable zone only
            caught.append(bool(ok.any())); leads.append(ti - te[ok].min() if ok.any() else np.nan)
            ev += count_events(te[flag & (te < ti - zone)]); minutes += max(ti - zone, 0) / 60   # early alarms = false alarms
        else:
            n = count_events(te[flag]); ev += n; minutes += Mx.dur.values[ix[0]] / 60
            adl_alarm += int(n > 0); n_adl += 1
    leads = np.array(leads, float)
    return {"strict_recall": float(np.mean(caught)), "median_lead_s": float(np.nanmedian(leads)) if np.isfinite(leads).any() else np.nan,
            "false_alarms_per_min": ev / max(minutes, 1e-9), "adl_trials_with_alarm": adl_alarm / max(n_adl, 1)}

DELTAS = [0.0, 0.2, 0.4]; EXT = []
for tau in [t for t in (0.0, 0.3, 0.6) if t in STORE]:
    lab, oof = STORE[tau]["lab"], STORE[tau]["oof"]; fold = get_fold_ids(Wm, tau)
    for delta in DELTAS:
        s_all = np.full(len(Wm), np.nan)
        for t, ix in tidx.items():
            if not np.isnan(oof[ix]).any(): s_all[ix] = smooth_causal(Wm.t_end.values[ix], oof[ix], delta)
        thr_row = np.full(len(Wm), np.nan)
        for k in range(N_SPLITS):
            tr = (fold != k) & (fold >= 0) & ~np.isnan(lab) & ~np.isnan(s_all)
            thr_row[fold == k] = best_threshold(lab[tr].astype(float), s_all[tr])
        EXT.append({"tau": tau, "smoothing_s": delta, **strict_metrics(Wm, tidx, s_all, thr_row, tau)})
ext_df = pd.DataFrame(EXT)
len_ = res[["tau", "fall_recall_trial", "adl_alarms_per_min"]].rename(columns={"fall_recall_trial": "lenient_recall (Sec.5)", "adl_alarms_per_min": "lenient_alarms/min (Sec.5)"})
ext_df = ext_df.merge(len_.assign(smoothing_s=0.0), on=["tau", "smoothing_s"], how="left")
ext_df.to_csv(OUT_DIR / "table_extension_strict_smoothing.csv", index=False); ext_df.round(3)

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
for tau in ext_df.tau.unique():
    d = ext_df[ext_df.tau == tau]
    axs[0].plot(d.smoothing_s, d.strict_recall, "o-", label=f"tau={tau:.1f}s"); axs[1].plot(d.smoothing_s, d.false_alarms_per_min, "o-", label=f"tau={tau:.1f}s")
axs[0].set_ylabel("STRICT falls caught"); axs[1].set_ylabel("false alarms per minute")
for a in axs: a.set_xlabel("persistence window Delta (s)"); a.legend()
axs[0].set_title("Recall under the strict metric"); axs[1].set_title("Persistence suppresses false alarms")
plt.tight_layout(); plt.savefig(OUT_DIR / "fig09_extension_persistence.png", dpi=150); plt.show()

def demo_trial(name_part, tau=0.3, delta=0.0):
    ti_ = next((i for i, r in enumerate(trials) if name_part in r["name"] and MAIN_POS in r["acc"]), None)
    if ti_ is None: print("trial not found:", name_part); return
    r = trials[ti_]; rows = np.where(Wm.trial.values == ti_)[0]; subj = r["subject"]
    lab, oof = STORE[tau]["lab"], STORE[tau]["oof"]; others = Wm.subject.values != subj
    ok = ~np.isnan(lab) & ~np.isnan(oof) & others
    thr = best_threshold(lab[ok].astype(float), oof[ok])                         # threshold from OTHER subjects only
    L = np.where(~np.isnan(lab) & others)[0]
    model = make_rf().fit(Wf.values[L], lab[L])                                  # trained without this person
    te = Wm.t_end.values[rows]; s = smooth_causal(te, model.predict_proba(Wf.values[rows])[:, 1], delta)
    alarm = np.where(s >= thr)[0]; smv = np.linalg.norm(r["acc"][MAIN_POS], axis=1)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True)
    ax[0].plot(np.arange(len(smv)) / FS, smv, color="#b2182b" if r["is_fall"] else "#2166ac"); ax[0].set_ylabel("chest SMV (g)")
    ax[1].plot(te, s, color="k"); ax[1].axhline(thr, ls="--", color="grey", label=f"alarm threshold {thr:.2f}"); ax[1].set_ylabel("model score"); ax[1].set_xlabel("time (s)")
    if r["is_fall"]:
        for a in ax: a.axvline(r["t_imp"], color="k", ls=":", lw=2)
    msg = "no alarm raised"
    if len(alarm):
        ta = te[alarm[0]]
        for a in ax: a.axvline(ta, color="orange", lw=2.5)
        msg = f"first alarm at {ta:.2f} s" + (f"  ->  {r['t_imp'] - ta:.2f} s BEFORE impact" if r["is_fall"] else "  (FALSE ALARM)")
    ax[0].set_title(f"{r['name'][:48]} | tau={tau}s | {msg}", fontsize=9); ax[1].legend(loc="upper left"); plt.tight_layout(); plt.show(); print(msg)

for nm in ["Subject_07_Fall_backwardFall_4", "Subject_04_ADL_Jogging_2", "Subject_07_ADL_Walking_2", "Subject_02_ADL_Sitting_GettingUpOnAChair_1"]:
    demo_trial(nm)