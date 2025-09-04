# -*- coding: utf-8 -*-

#One?Day XGBoost Trainer PRO (修复+适配分割命名+回退 ALL_* 版)
#----------------------------------------------------------
#更新点：
#1) 兼容你的分割命名：支持 `train_top{pid}.csv` / `val_top{pid}.csv`；
#2) 当找不到 per?PID split 文件时，自动**回退到 ALL_TRAIN/ALL_VAL** 按 ProgramID 过滤训练；
#3) ProgramID 列名兼容：优先使用 `ProgramID`，若无则使用 `ProgramID_encoded`；
#4) 保留上一版修复（FixedTrial 报错）与所有“质量护栏”。

#目标：让 oneday?pro 在你当前的分割产物下**真正覆盖每个 PID**，而不是只训练 Others。

from __future__ import annotations

import os, json, time, warnings, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from optuna.integration import XGBoostPruningCallback
from joblib import Parallel, delayed, dump

from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ============================= CONFIG ============================= #
SPLIT_DIR   = "/home/master/wzheng/projects/model_training/data/top40_splits"
TOPN_JSON   = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"

MODEL_DIR   = "/home/master/wzheng/projects/model_training/models/v5.4_cls_xgb_oneday_pro"
REPORT_DIR  = "/home/master/wzheng/projects/model_training/reports/v5.4_cls_xgb_oneday_pro"
FEATURES_JSON = "/home/master/wzheng/projects/model_training/models/v5.4_cls_features.json"

ALL_TRAIN   = "/home/master/wzheng/projects/model_training/data/40train.csv"
ALL_VAL     = "/home/master/wzheng/projects/model_training/data/40val.csv"

RANDOM_SEED = 42

# —— One?Day knobs ——
N_TRIALS_MAX            = 40      # study trial 上限（时间预算会提前截断）
PID_TIME_BUDGET_MIN     = 30      # 每个 PID 的总搜索预算（分钟）
EARLY_STOP_ROUNDS       = 50
N_ESTIMATORS_MAX        = 800
THREADS_PER_MODEL       = 2
CONCURRENCY             = 6
USE_GPU                 = False   # 若有 GPU 版 xgboost，可改 True

# 质量护栏
SNAPSHOT_SEEDS          = [42, 2025, 13579]  # 多种子快照集成
ENABLE_SNAPSHOT_ENSEMBLE= True
ENABLE_THRESHOLD_TUNING = True               # 二分类阈值微调
ENABLE_PLATT_CALIB      = True               # 仅二分类 + 失衡时启用
IMBALANCE_RATIO         = 0.15               # minority/majority < 0.15 视作失衡

# 难点 PID（可单独加时）
HARD_PID_EXTRA_MINUTES  = 10
HARD_PIDS               = set()              # e.g., {"pidA","pidB"}

TARGET_COL  = "BucketLabel"
TIME_COL    = "SubmitTime"
REG_Y_COL   = "RemoteWallClockTime"
# ProgramID 兼容两种列名
PROGRAM_COL_CANDIDATES = ("ProgramID", "ProgramID_encoded")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 预加载 ALL_*（用于 Others & 缺分割时的回退）
ALL_TR_DF = pd.read_csv(ALL_TRAIN) if os.path.exists(ALL_TRAIN) else pd.DataFrame()
ALL_VA_DF = pd.read_csv(ALL_VAL) if os.path.exists(ALL_VAL) else pd.DataFrame()

# ============================== Utils ============================= #

def _get_program_col(df: pd.DataFrame) -> str:
    for c in PROGRAM_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise KeyError(f"Neither ProgramID nor ProgramID_encoded present in columns: {list(df.columns)[:10]}")


def _numeric_features(df: pd.DataFrame, drop_cols: Tuple[str, ...]) -> List[str]:
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols].select_dtypes(include=[np.number]).columns.tolist()


def _maybe_save_features(cols: List[str]):
    try:
        if not os.path.exists(FEATURES_JSON):
            with open(FEATURES_JSON, "w", encoding="utf-8") as f:
                json.dump({"feature_cols": cols}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _load_top_pids() -> List[str]:
    with open(TOPN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "top" in data: pids = list(map(str, data["top"]))
        elif "programids" in data: pids = list(map(str, data["programids"]))
        else: pids = list(map(str, data))
    else:
        pids = list(map(str, data))
    return pids


def _pid_split_paths(pid: str) -> Tuple[str, str]:
#    兼容多种命名：
#    - {pid}_train.csv / {pid}_val.csv
#    - {pid}-train.csv / {pid}-val.csv
#    - {pid}/train.csv  / {pid}/val.csv
#    - train_top{pid}.csv / val_top{pid}.csv   ← 你当前的命名
    
    candidates = [
        (os.path.join(SPLIT_DIR, f"{pid}_train.csv"), os.path.join(SPLIT_DIR, f"{pid}_val.csv")),
        (os.path.join(SPLIT_DIR, f"{pid}-train.csv"), os.path.join(SPLIT_DIR, f"{pid}-val.csv")),
        (os.path.join(SPLIT_DIR, pid, "train.csv"), os.path.join(SPLIT_DIR, pid, "val.csv")),
        (os.path.join(SPLIT_DIR, f"train_top{pid}.csv"), os.path.join(SPLIT_DIR, f"val_top{pid}.csv")),
    ]
    for tr, va in candidates:
        if os.path.exists(tr) and os.path.exists(va):
            return tr, va
    return "", ""


def _class_weights(y: np.ndarray) -> Optional[np.ndarray]:
    classes = np.unique(y)
    if classes.size < 2:
        return None
    weights = compute_class_weight("balanced", classes=classes, y=y)
    weight_map = {c: w for c, w in zip(classes, weights)}
    return np.vectorize(weight_map.get)(y)


def _is_binary(y: np.ndarray) -> bool:
    return np.unique(y).size == 2


def _is_imbalanced(y: np.ndarray) -> bool:
    vals, cnts = np.unique(y, return_counts=True)
    if len(cnts) < 2: return False
    r = cnts.min() / cnts.max()
    return r < IMBALANCE_RATIO


def _tune_threshold_binary(y_true: np.ndarray, p1: np.ndarray) -> Tuple[float, float, float]:
    best_thr, best_f1, best_acc = 0.5, -1.0, 0.0
    for step, rng in [(0.05, np.arange(0.05,0.96,0.05)), (0.01, None)]:
        candidates = rng if rng is not None else np.arange(max(0.05,best_thr-0.05), min(0.95,best_thr+0.05)+1e-9, 0.01)
        for t in candidates:
            pred = (p1 >= t).astype(int)
            f1m = f1_score(y_true, pred, average="macro")
            acc = accuracy_score(y_true, pred)
            if f1m > best_f1 or (math.isclose(f1m,best_f1,rel_tol=1e-9) and acc>best_acc):
                best_thr, best_f1, best_acc = float(t), float(f1m), float(acc)
    return best_thr, best_f1, best_acc


def _platt_calibrate(p1: np.ndarray, y_true: np.ndarray) -> Tuple[np.ndarray, LogisticRegression]:
    eps = 1e-6
    p1c = np.clip(p1, eps, 1-eps)
    X = np.log(p1c / (1 - p1c)).reshape(-1,1)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, y_true)
    p1_cal = lr.predict_proba(X)[:,1]
    return p1_cal, lr

# =========================== Core Training ======================== #
@dataclass
class TrainResult:
    pid: str
    train_size: int
    val_size: int
    accuracy: Optional[float]
    f1_macro: Optional[float]
    best_params: Optional[Dict]
    model_paths: List[str]
    note: str = ""
    threshold: Optional[float] = None  # for binary
    calibrated: bool = False


def _base_params(n_classes: int, objective: str, seed: int) -> Dict:
    tree_method = "gpu_hist" if USE_GPU else "hist"
    p = {
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "n_estimators": N_ESTIMATORS_MAX,
        "early_stopping_rounds": EARLY_STOP_ROUNDS,
        "n_jobs": THREADS_PER_MODEL,
        "tree_method": tree_method,
        "random_state": seed,
        "objective": objective,
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "verbosity": 0,
    }
    if n_classes > 2:
        p["num_class"] = n_classes
    return p


def _build_model_params(trial: optuna.Trial, n_classes: int, objective: str, seed: int) -> Dict:
    tree_method = "gpu_hist" if USE_GPU else "hist"
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "n_estimators": N_ESTIMATORS_MAX,
        "early_stopping_rounds": EARLY_STOP_ROUNDS,
        "n_jobs": THREADS_PER_MODEL,
        "tree_method": tree_method,
        "random_state": seed,
        "objective": objective,
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "verbosity": 0,
    }
    if n_classes > 2:
        params["num_class"] = n_classes
    return params


def _fit_one_seed(Xtr, ytr, Xva, yva, pid: str, seed: int, timeout_s: int):
    n_classes = len(np.unique(yva))
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"

    def obj(trial: optuna.Trial) -> float:
        params = _build_model_params(trial, n_classes, objective, seed)
        model = XGBClassifier(**params)
        sw = _class_weights(ytr)
        metric_name = "validation_0-mlogloss" if n_classes > 2 else "validation_0-logloss"
        callbacks = [XGBoostPruningCallback(trial, metric_name)]
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False,
            sample_weight=sw,
            callbacks=callbacks,
        )
        preds = model.predict(Xva)
        f1m = f1_score(yva, preds, average="macro")
        return 1.0 - f1m

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=50, interval_steps=25)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(obj, n_trials=N_TRIALS_MAX, timeout=timeout_s, gc_after_trial=True, show_progress_bar=False)

    best = study.best_params if len(study.trials) > 0 else {}

    fixed = {
        "n_estimators": N_ESTIMATORS_MAX,
        "early_stopping_rounds": EARLY_STOP_ROUNDS,
        "n_jobs": THREADS_PER_MODEL,
        "tree_method": ("gpu_hist" if USE_GPU else "hist"),
        "random_state": seed,
        "verbosity": 0,
    }
    best_full = _base_params(n_classes, objective, seed)
    if "lr" in best and "learning_rate" not in best:
        best["learning_rate"] = best.pop("lr")
    best_full.update(best)
    best_full.update(fixed)

    final = XGBClassifier(**best_full)
    sw = _class_weights(ytr)
    final.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False, sample_weight=sw)

    if n_classes > 2:
        p = final.predict_proba(Xva)
    else:
        p = final.predict_proba(Xva)[:,1]

    return best_full, final, p


def _numeric_block(df_tr, df_va):
    # 动态识别 ProgramID 列，便于从特征中排除
    prog_col = _get_program_col(df_tr)
    drop_cols = (TARGET_COL, TIME_COL, REG_Y_COL, prog_col)
    feat_cols = _numeric_features(df_tr, drop_cols)
    _maybe_save_features(feat_cols)
    Xtr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    ytr = df_tr[TARGET_COL].to_numpy()
    Xva = df_va[feat_cols].to_numpy(dtype=np.float32)
    yva = df_va[TARGET_COL].to_numpy()
    return Xtr, ytr, Xva, yva


def _fit_with_guardrails(Xtr, ytr, Xva, yva, pid: str, total_budget_min: int) -> TrainResult:
    t0 = time.time()
    if np.unique(ytr).size < 2 or np.unique(yva).size < 2:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(Xtr, ytr)
        acc = accuracy_score(yva, dummy.predict(Xva))
        f1m = f1_score(yva, dummy.predict(Xva), average="macro") if np.unique(yva).size>1 else None
        mpath = os.path.join(MODEL_DIR, f"dummy_pid_{pid}.joblib")
        dump(dummy, mpath)
        return TrainResult(pid, len(ytr), len(yva), float(acc) if acc is not None else None,
                           float(f1m) if f1m is not None else None, {}, [mpath], note="single class")

    seeds = SNAPSHOT_SEEDS if ENABLE_SNAPSHOT_ENSEMBLE else [RANDOM_SEED]
    per_seed_timeout = max(60, int((total_budget_min * 60) / len(seeds)))

    all_models, all_probs, best_params_list = [], [], []
    for s in seeds:
        best, model, proba = _fit_one_seed(Xtr, ytr, Xva, yva, pid, seed=s, timeout_s=per_seed_timeout)
        all_models.append(model)
        all_probs.append(proba)
        best_params_list.append(best)

    if _is_binary(yva):
        p_ens = np.mean(np.vstack(all_probs), axis=0)
    else:
        p_ens = np.mean(np.stack(all_probs, axis=0), axis=0)

    calibrated = False
    thr = None

    if _is_binary(yva):
        if ENABLE_PLATT_CALIB and _is_imbalanced(yva):
            p_cal, lr = _platt_calibrate(p_ens, yva)
            p_ens = p_cal
            calibrated = True
        if ENABLE_THRESHOLD_TUNING:
            thr, best_f1, best_acc = _tune_threshold_binary(yva, p_ens)
            y_pred = (p_ens >= thr).astype(int)
        else:
            y_pred = (p_ens >= 0.5).astype(int)
        acc = accuracy_score(yva, y_pred)
        f1m = f1_score(yva, y_pred, average="macro")
    else:
        y_pred = np.argmax(p_ens, axis=1)
        acc = accuracy_score(yva, y_pred)
        f1m = f1_score(yva, y_pred, average="macro")

    mpaths = []
    for i, m in enumerate(all_models):
        mpath = os.path.join(MODEL_DIR, f"xgb_pid_{pid}_seed{i}.joblib")
        dump(m, mpath)
        mpaths.append(mpath)

    note = f"elapsed {int(time.time()-t0)}s; seeds={len(seeds)}; calib={calibrated}; thr={thr}"
    best_rep = best_params_list[0] if best_params_list else {}

    return TrainResult(pid, len(ytr), len(yva), float(acc), float(f1m), best_rep, mpaths, note=note, threshold=thr, calibrated=calibrated)


# =========================== Pipeline per PID ===================== #

def _load_pid_frames(pid: str) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
#    返回 (df_tr, df_va, source)；source in {"split","fallback","empty"}
    tr_path, va_path = _pid_split_paths(pid)
    if tr_path and va_path:
        return pd.read_csv(tr_path), pd.read_csv(va_path), "split"
    # 回退到 ALL_* 过滤
    if not ALL_TR_DF.empty and not ALL_VA_DF.empty:
        # 选可用的 ProgramID 列
        prog_col_tr = _get_program_col(ALL_TR_DF)
        prog_col_va = _get_program_col(ALL_VA_DF)
        pid_str = str(pid)
        df_tr = ALL_TR_DF[ALL_TR_DF[prog_col_tr].astype(str) == pid_str].copy()
        df_va = ALL_VA_DF[ALL_VA_DF[prog_col_va].astype(str) == pid_str].copy()
        if not df_tr.empty or not df_va.empty:
            return df_tr, df_va, "fallback"
    return pd.DataFrame(), pd.DataFrame(), "empty"


def _train_one_pid(pid: str) -> TrainResult:
    df_tr, df_va, source = _load_pid_frames(pid)
    if df_tr.empty and df_va.empty:
        print(f"[SKIP] {pid}: no split & no rows in ALL_*; skip.")
        return TrainResult(pid, 0, 0, None, None, None, [], note="no data")

    print(f"[PID {pid}] source={source} | train={len(df_tr)} val={len(df_va)} | classes={df_tr[TARGET_COL].nunique() if TARGET_COL in df_tr else 'NA'}/{df_va[TARGET_COL].nunique() if TARGET_COL in df_va else 'NA'}")

    # 若某边为空，尽量避免报错（XGB 需要都存在）；空的那边会触发 Dummy/跳过
    if TARGET_COL not in df_tr.columns or TARGET_COL not in df_va.columns:
        print(f"[SKIP] {pid}: missing target in train/val; skip.")
        return TrainResult(pid, len(df_tr), len(df_va), None, None, None, [], note="missing target")

    Xtr, ytr, Xva, yva = _numeric_block(df_tr, df_va)
    budget = PID_TIME_BUDGET_MIN + (HARD_PID_EXTRA_MINUTES if pid in HARD_PIDS else 0)
    return _fit_with_guardrails(Xtr, ytr, Xva, yva, pid, total_budget_min=budget)


# =========================== Others Group ========================= #

def _train_others(top_pids: List[str]) -> TrainResult:
    if ALL_TR_DF.empty or ALL_VA_DF.empty:
        return TrainResult("Others", 0, 0, None, None, None, [], note="missing ALL_TRAIN/ALL_VAL")

    prog_col_tr = _get_program_col(ALL_TR_DF)
    prog_col_va = _get_program_col(ALL_VA_DF)
    top_set = set(map(str, top_pids))

    df_tr = ALL_TR_DF[~ALL_TR_DF[prog_col_tr].astype(str).isin(top_set)].copy()
    df_va = ALL_VA_DF[~ALL_VA_DF[prog_col_va].astype(str).isin(top_set)].copy()

    if df_tr.empty or df_va.empty:
        return TrainResult("Others", len(df_tr), len(df_va), None, None, None, [], note="insufficient data")

    Xtr, ytr, Xva, yva = _numeric_block(df_tr, df_va)
    budget = PID_TIME_BUDGET_MIN + (HARD_PID_EXTRA_MINUTES if "Others" in HARD_PIDS else 0)
    return _fit_with_guardrails(Xtr, ytr, Xva, yva, pid="Others", total_budget_min=budget)


# =============================== Main ============================ #

def main():
    print("[One?Day PRO ? split?aware] XGB Trainer starting")
    print(f"Budget: {PID_TIME_BUDGET_MIN} min / PID | Concurrency: {CONCURRENCY} | Threads/Model: {THREADS_PER_MODEL}")
    print(f"Snapshot: {ENABLE_SNAPSHOT_ENSEMBLE} seeds={len(SNAPSHOT_SEEDS)} | ThrTuning: {ENABLE_THRESHOLD_TUNING} | Platt: {ENABLE_PLATT_CALIB}")

    top_pids = _load_top_pids()
    print(f"Loaded top ProgramIDs: {len(top_pids)}")

    results: List[TrainResult] = Parallel(n_jobs=CONCURRENCY, prefer="processes")(
        delayed(_train_one_pid)(pid) for pid in top_pids
    )

    print("[Stage] Training Others")
    res_others = _train_others(top_pids)
    results.append(res_others)

    rows = []
    for r in results:
        rows.append({
            "PID": r.pid,
            "TrainSize": r.train_size,
            "ValSize": r.val_size,
            "Accuracy": r.accuracy,
            "F1_macro": r.f1_macro,
            "BestParams": json.dumps(r.best_params) if isinstance(r.best_params, dict) else None,
            "ModelPaths": json.dumps(r.model_paths),
            "Threshold": r.threshold,
            "Calibrated": r.calibrated,
            "Note": r.note,
        })
    sum_path = os.path.join(REPORT_DIR, "v5.4_cls_xgb_oneday_pro_summary.csv")
    os.makedirs(REPORT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(sum_path, index=False)
    print("All models finished. Summary ->", sum_path)


if __name__ == "__main__":
    main()
