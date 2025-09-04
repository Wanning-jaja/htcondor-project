# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, sys, warnings
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import optuna
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

# ------------------ �ɵ� CONFIG ------------------
SPLIT_DIR   = "/home/master/wzheng/projects/model_training/data/top40_splits"
TOPN_JSON   = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"

MODEL_DIR   = "/home/master/wzheng/projects/model_training/models/v5.5_cls_xgb_upspeed"
REPORT_DIR  = "/home/master/wzheng/projects/model_training/reports/v5.5_cls_xgb_upspeed"
FEATURES_JSON = "/home/master/wzheng/projects/model_training/models/v5.4_cls_features.json"

ALL_TRAIN   = "/home/master/wzheng/projects/model_training/data/40train.csv"
ALL_VAL     = "/home/master/wzheng/projects/model_training/data/40val.csv"

RANDOM_SEED = 42

TARGET_COL  = "BucketLabel"          # ����Ŀ��
TIME_COL    = "SubmitTime"
REG_Y_COL   = "RemoteWallClockTime"  # ������ѵ��

# �ֵ������������CPU-only ���飩
SMALL_MAX_ROWS = 50_000
MEDIUM_MAX_ROWS = 300_000
MIN_CLASS_PER_LABEL = 200
OPTUNA_TRAIN_CAP = 200_000  # �����������������ڵ��ε�����
OPTUNA_TIMEOUT = int(os.getenv("OPTUNA_TIMEOUT", "0"))  # ÿ�� block �Ѳγ�ʱ(��)��0 ��ʾ����

# ÿ�� trial ��
N_TRIALS_SMALL = 0      # С�������Ѳ�
N_TRIALS_MED   = 10
N_TRIALS_LARGE = 15

# ͳһ�����CPU��
N_JOBS = int(os.getenv("SLURM_CPUS_PER_TASK", os.getenv("N_JOBS", "12")))
FAST_XGB_BASE = dict(
    tree_method="hist",
    n_jobs=N_JOBS,
    max_bin=128,
)

# ��־���루���迪����
# warnings.filterwarnings("ignore", category=UserWarning)
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# �����û�����������·��/����
SPLIT_DIR   = os.getenv("SPLIT_DIR", SPLIT_DIR)
TOPN_JSON   = os.getenv("TOPN_JSON", TOPN_JSON)
MODEL_DIR   = os.getenv("MODEL_DIR", MODEL_DIR)
REPORT_DIR  = os.getenv("REPORT_DIR", REPORT_DIR)
ALL_TRAIN   = os.getenv("ALL_TRAIN", ALL_TRAIN)
ALL_VAL     = os.getenv("ALL_VAL", ALL_VAL)

SMALL_MAX_ROWS       = int(os.getenv("SMALL_MAX_ROWS", SMALL_MAX_ROWS))
MEDIUM_MAX_ROWS      = int(os.getenv("MEDIUM_MAX_ROWS", MEDIUM_MAX_ROWS))
MIN_CLASS_PER_LABEL  = int(os.getenv("MIN_CLASS_PER_LABEL", MIN_CLASS_PER_LABEL))
OPTUNA_TRAIN_CAP     = int(os.getenv("OPTUNA_TRAIN_CAP", OPTUNA_TRAIN_CAP))
N_TRIALS_SMALL       = int(os.getenv("N_TRIALS_SMALL", N_TRIALS_SMALL))
N_TRIALS_MED         = int(os.getenv("N_TRIALS_MED", N_TRIALS_MED))
N_TRIALS_LARGE       = int(os.getenv("N_TRIALS_LARGE", N_TRIALS_LARGE))

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ------------------ Utils ------------------
def _numeric_features(df: pd.DataFrame, drop_cols: Tuple[str, ...]) -> List[str]:
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols].select_dtypes(include=[np.number]).columns.tolist()

def _maybe_save_features(cols: List[str]):
    try:
        if not os.path.exists(FEATURES_JSON):
            with open(FEATURES_JSON, "w", encoding="utf-8") as f:
                json.dump({"feature_cols": cols}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[WARN] save features.json failed:", e)

def _xgb_objective_and_extras(n_classes: int):
    if n_classes <= 1:
        raise ValueError("n_classes must be >= 2 for XGBoost training.")
    if n_classes == 2:
        return "binary:logistic", {}
    else:
        return "multi:softprob", {"num_class": n_classes}

def _decide_bucket(n_rows: int, y: np.ndarray) -> str:
    min_cls = pd.Series(y).value_counts().min() if len(y) else 0
    if (n_rows < SMALL_MAX_ROWS) or (min_cls < MIN_CLASS_PER_LABEL):
        return "small"
    elif n_rows <= MEDIUM_MAX_ROWS:
        return "medium"
    else:
        return "large"

def stratified_cap_sample(X: np.ndarray, y: np.ndarray, cap: int, random_state: int = RANDOM_SEED):
    n = len(y)
    if n <= cap:
        return X, y
    test_size = cap / float(n)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for _, idx in sss.split(X, y):
        return X[idx], y[idx]

def _fit_with_optuna(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
                     n_classes: int, class_weights=None, n_trials: int = 10) -> Dict[str, Any]:
    from optuna.pruners import MedianPruner
    from optuna.integration import XGBoostPruningCallback

    objective, extra = _xgb_objective_and_extras(n_classes)

    def obj(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 0.5),

            "n_estimators": 1000,
            "random_state": RANDOM_SEED,
            "objective": objective,
            "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
            "verbosity": 0,
            **extra,
            **FAST_XGB_BASE,
        }

        # �Ѽ�֦�ص��ŵ������������� deprecation ����
        pruning_cb = XGBoostPruningCallback(trial, "validation_0-" + params["eval_metric"])
        model = XGBClassifier(**params, callbacks=[pruning_cb])
        model.set_params(early_stopping_rounds=50)

        fit_kwargs = {"eval_set": [(Xva, yva)], "verbose": False}
        if class_weights is not None:
            # �� class_weights -> sample_weight
            uniq = np.unique(ytr)
            wmap = {int(c): float(w) for c, w in zip(uniq, compute_class_weight("balanced", classes=uniq, y=ytr))}
            sw_tr = np.asarray([wmap[int(c)] for c in ytr], dtype=float)
            fit_kwargs["sample_weight"] = sw_tr

        model.fit(Xtr, ytr, **fit_kwargs)

        pred = model.predict(Xva)
        return -f1_score(yva, pred, average="macro", zero_division=0)

    study = optuna.create_study(direction="minimize", pruner=MedianPruner(n_warmup_steps=10))
    n_trials = max(1, int(n_trials))
    study.optimize(obj, n_trials=n_trials, timeout=(OPTUNA_TIMEOUT or None))

    best = study.best_params
    best.update({
        "n_estimators": 1000,
        "random_state": RANDOM_SEED,
        "objective": objective,
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "verbosity": 0,
        **extra,
        **FAST_XGB_BASE,
    })
    return best

def _default_params(n_classes: int) -> Dict[str, Any]:
    obj, extra = _xgb_objective_and_extras(n_classes)
    params = {
        "learning_rate": 0.08,
        "max_depth": 6,
        "min_child_weight": 2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.2,
        "n_estimators": 600,
        "random_state": RANDOM_SEED,
        "objective": obj,
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "verbosity": 0,
        **extra,
        **FAST_XGB_BASE,
    }
    return params

# ------------------ ������ ------------------
def train_block(df_tr: pd.DataFrame, df_va: pd.DataFrame, pid_label: str,
                model_prefix: str, summary_rows: list):
    feat_cols = _numeric_features(df_tr, drop_cols=(TARGET_COL, TIME_COL, REG_Y_COL))
    if not feat_cols:
        print(f"[WARN] {pid_label}: no numeric features left.")
        # ����Ҳ��¼
        summary_rows.append({
            "ProgramID_encoded": pid_label,
            "TrainSize": len(df_tr),
            "ValSizeOriginal": len(df_va),
            "ValDroppedUnseen": 0,
            "ValSizeUsed": 0,
            "Accuracy": np.nan,
            "F1_macro": np.nan,
            "BestParams": json.dumps({"reason": "no_numeric_features"}),
            "ModelPath": "",
            "Status": "ERROR_NO_FEATURES",
        })
        return

    _maybe_save_features(feat_cols)

    if len(df_va) == 0:
        print(f"[WARN] {pid_label}: no validation split, skip.")
        summary_rows.append({
            "ProgramID_encoded": pid_label,
            "TrainSize": len(df_tr),
            "ValSizeOriginal": 0,
            "ValDroppedUnseen": 0,
            "ValSizeUsed": 0,
            "Accuracy": np.nan,
            "F1_macro": np.nan,
            "BestParams": json.dumps({"reason": "no_val_split"}),
            "ModelPath": "",
            "Status": "SKIPPED_NO_VAL",
        })
        return

    # numpy float32
    Xtr = df_tr[feat_cols].to_numpy(dtype=np.float32)
    ytr = df_tr[TARGET_COL].astype(int).to_numpy()
    Xva = df_va[feat_cols].to_numpy(dtype=np.float32)
    yva = df_va[TARGET_COL].astype(int).to_numpy()

    # ---- ������֤����ѵ��δ���ֵı�ǩ�������������� ----
    classes_tr = np.unique(ytr)
    val_size_original = len(yva)
    va_mask = np.isin(yva, classes_tr)
    val_dropped = int((~va_mask).sum())
    if val_dropped > 0:
        unseen = np.unique(yva[~va_mask])
        print(f"[WARN] {pid_label}: drop {val_dropped} val rows with unseen labels {unseen} not in train classes {classes_tr}.")
        Xva = Xva[va_mask]
        yva = yva[va_mask]

    if len(yva) == 0:
        print(f"[WARN] {pid_label}: all val rows dropped after filtering unseen labels, skip.")
        summary_rows.append({
            "ProgramID_encoded": pid_label,
            "TrainSize": len(df_tr),
            "ValSizeOriginal": val_size_original,
            "ValDroppedUnseen": val_dropped,
            "ValSizeUsed": 0,
            "Accuracy": np.nan,
            "F1_macro": np.nan,
            "BestParams": json.dumps({"reason": "val_all_dropped"}),
            "ModelPath": "",
            "Status": "SKIPPED_VAL_EMPTY",
        })
        return

    # ����ǩӳ��Ϊ���� 0..K-1�����ף�
    label2new = {old: i for i, old in enumerate(sorted(classes_tr))}
    ytr = np.array([label2new[v] for v in ytr], dtype=int)
    yva = np.array([label2new[v] for v in yva], dtype=int)
    classes = np.arange(len(classes_tr))
    n_classes = len(classes)

    # ���ࣺ��Ԥ��
    if n_classes == 1:
        const_label = 0  # ӳ���Ψһ�༴ 0
        final = DummyClassifier(strategy="constant", constant=const_label)
        final.fit(Xtr, ytr)
        pred = final.predict(Xva)
        acc  = accuracy_score(yva, pred)
        f1m  = f1_score(yva, pred, average="macro", zero_division=0)

        mpath = os.path.join(MODEL_DIR, f"{model_prefix}_constant.joblib")
        dump(final, mpath)

        rpt = classification_report(yva, pred, output_dict=True, zero_division=0)
        pd.DataFrame(rpt).to_csv(os.path.join(REPORT_DIR, f"{pid_label}_classification_report.csv"))

        summary_rows.append({
            "ProgramID_encoded": pid_label,
            "TrainSize": len(df_tr),
            "ValSizeOriginal": val_size_original,
            "ValDroppedUnseen": val_dropped,
            "ValSizeUsed": len(yva),
            "Accuracy": acc,
            "F1_macro": f1m,
            "BestParams": json.dumps({"model": "constant", "label_mapped": const_label}),
            "ModelPath": mpath,
            "Status": "SINGLE_CLASS_OK",
        })
        print(f"{pid_label} | SINGLE-CLASS -> const={const_label} | Acc={acc:.3f} | F1_macro={f1m:.3f}")
        return

    # ��/����
    bucket = _decide_bucket(len(df_tr), ytr)
    if bucket == "small":
        n_trials = N_TRIALS_SMALL
    elif bucket == "medium":
        n_trials = N_TRIALS_MED
    else:
        n_trials = N_TRIALS_LARGE

    print(f"{pid_label} | bucket={bucket} | Train={len(df_tr)} | Val={len(yva)} | classes={n_classes} | trials={n_trials}")

    # class weights������ѵ����ȫ����
    cls_weights_full = compute_class_weight("balanced", classes=classes, y=ytr)

    if n_trials <= 0:
        best = _default_params(n_classes)
    else:
        # ����������
        Xcap, ycap = stratified_cap_sample(Xtr, ytr, OPTUNA_TRAIN_CAP, random_state=RANDOM_SEED)
        best = _fit_with_optuna(Xcap, ycap, Xva, yva, n_classes=n_classes,
                                class_weights=True, n_trials=n_trials)

    # ����ȫ����ѵ������ͣ��
    final = XGBClassifier(**best).set_params(early_stopping_rounds=50)
    wmap = {int(c): float(w) for c, w in zip(classes, cls_weights_full)}
    sw_tr = np.asarray([wmap[int(c)] for c in ytr], dtype=float)

    final.fit(Xtr, ytr, sample_weight=sw_tr, eval_set=[(Xva, yva)], verbose=False)
    pred = final.predict(Xva)
    acc  = accuracy_score(yva, pred)
    f1m  = f1_score(yva, pred, average="macro", zero_division=0)

    mpath = os.path.join(MODEL_DIR, f"{model_prefix}_optuna.joblib" if n_trials > 0 else f"{model_prefix}_default.joblib")
    dump(final, mpath)

    rpt = classification_report(yva, pred, output_dict=True, zero_division=0)
    pd.DataFrame(rpt).to_csv(os.path.join(REPORT_DIR, f"{pid_label}_classification_report.csv"))

    summary_rows.append({
        "ProgramID_encoded": pid_label,
        "TrainSize": len(df_tr),
        "ValSizeOriginal": val_size_original,
        "ValDroppedUnseen": val_dropped,
        "ValSizeUsed": len(yva),
        "Accuracy": acc,
        "F1_macro": f1m,
        "BestParams": json.dumps(best),
        "ModelPath": mpath,
        "Status": "OK",
    })
    print(f"{pid_label} | Acc={acc:.3f} | F1_macro={f1m:.3f}")

# ------------------ Main ------------------
def main():
    with open(TOPN_JSON, "r", encoding="utf-8") as f:
        top_programids = json.load(f)

    summary_rows = []

    for pid in top_programids:
        try:
            train_path = os.path.join(SPLIT_DIR, f"train_top{pid}.csv")
            val_path   = os.path.join(SPLIT_DIR, f"val_top{pid}.csv")

            if not os.path.exists(train_path):
                print(f"[SKIP] PID {pid}: train file not found")
                summary_rows.append({
                    "ProgramID_encoded": str(pid),
                    "TrainSize": 0,
                    "ValSizeOriginal": 0,
                    "ValDroppedUnseen": 0,
                    "ValSizeUsed": 0,
                    "Accuracy": np.nan,
                    "F1_macro": np.nan,
                    "BestParams": json.dumps({"reason": "train_file_missing"}),
                    "ModelPath": "",
                    "Status": "SKIPPED_NO_TRAIN",
                })
                continue

            df_tr = pd.read_csv(train_path, low_memory=False).dropna(subset=[TARGET_COL])
            if os.path.exists(val_path):
                df_va = pd.read_csv(val_path, low_memory=False).dropna(subset=[TARGET_COL])
            else:
                df_va = pd.DataFrame(columns=df_tr.columns)

            train_block(df_tr, df_va, pid_label=str(pid), model_prefix=f"xgb_cls_pid{pid}", summary_rows=summary_rows)

        except Exception as e:
            print(f"[ERROR] PID {pid} failed: {e}")
            summary_rows.append({
                "ProgramID_encoded": str(pid),
                "TrainSize": int(df_tr.shape[0]) if 'df_tr' in locals() else 0,
                "ValSizeOriginal": int(df_va.shape[0]) if 'df_va' in locals() else 0,
                "ValDroppedUnseen": np.nan,
                "ValSizeUsed": np.nan,
                "Accuracy": np.nan,
                "F1_macro": np.nan,
                "BestParams": json.dumps({"error": str(e)}),
                "ModelPath": "",
                "Status": "ERROR",
            })
            continue

    print("\n>> Training (Others) block ")
    try:
        df_all_tr = pd.read_csv(ALL_TRAIN, low_memory=False)
        df_all_va = pd.read_csv(ALL_VAL, low_memory=False)
        with open(TOPN_JSON, "r", encoding="utf-8") as f:
            top_set = set(json.load(f))

        df_ot_tr = df_all_tr[~df_all_tr["ProgramID_encoded"].isin(top_set)].dropna(subset=[TARGET_COL])
        df_ot_va = df_all_va[~df_all_va["ProgramID_encoded"].isin(top_set)].dropna(subset=[TARGET_COL])

        if len(df_ot_tr) and len(df_ot_va):
            train_block(df_ot_tr, df_ot_va, pid_label="Others", model_prefix="xgb_cls_others", summary_rows=summary_rows)
        else:
            print("[SKIP] Others: insufficient data.")
            summary_rows.append({
                "ProgramID_encoded": "Others",
                "TrainSize": int(df_ot_tr.shape[0]) if 'df_ot_tr' in locals() else 0,
                "ValSizeOriginal": int(df_ot_va.shape[0]) if 'df_ot_va' in locals() else 0,
                "ValDroppedUnseen": 0,
                "ValSizeUsed": 0,
                "Accuracy": np.nan,
                "F1_macro": np.nan,
                "BestParams": json.dumps({"reason": "insufficient_data"}),
                "ModelPath": "",
                "Status": "SKIPPED_OTHERS",
            })
    except Exception as e:
        print(f"[ERROR] Others block failed: {e}")
        summary_rows.append({
            "ProgramID_encoded": "Others",
            "TrainSize": 0,
            "ValSizeOriginal": 0,
            "ValDroppedUnseen": np.nan,
            "ValSizeUsed": np.nan,
            "Accuracy": np.nan,
            "F1_macro": np.nan,
            "BestParams": json.dumps({"error": str(e)}),
            "ModelPath": "",
            "Status": "ERROR_OTHERS",
        })

    sum_path = os.path.join(REPORT_DIR, "v5.5_cls_xgb_summary.csv")
    pd.DataFrame(summary_rows).to_csv(sum_path, index=False)
    print("\nAll XGB classification models finished. Summary ->", sum_path)

if __name__ == "__main__":
    main()
