# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, sys
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import optuna
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

# ---- 静音 Optuna 的失败 trial 日志（不影响搜参）----
optuna.logging.set_verbosity(optuna.logging.INFO)

# ====== 读取可选的单 PID 环境变量 ======
SINGLE_PID = os.getenv("SINGLE_PID")
if SINGLE_PID is not None:
    try:
        SINGLE_PID = int(SINGLE_PID)
    except Exception:
        print(f"[WARN] Bad SINGLE_PID={SINGLE_PID}, ignore.")
        SINGLE_PID = None

# ============== CONFIG ==============
SPLIT_DIR   = "/home/master/wzheng/projects/model_training/data/top40_splits"
TOPN_JSON   = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"

MODEL_DIR   = "/home/master/wzheng/projects/model_training/models/v5.4_cls_xgb"
REPORT_DIR  = "/home/master/wzheng/projects/model_training/reports/v5.4_cls_xgb"
FEATURES_JSON = "/home/master/wzheng/projects/model_training/models/v5.4_cls_features.json"

ALL_TRAIN   = "/home/master/wzheng/projects/model_training/data/40train.csv"
ALL_VAL     = "/home/master/wzheng/projects/model_training/data/40val.csv"

RANDOM_SEED = 42
N_TRIALS    = 50

TARGET_COL  = "BucketLabel"          # 分类目标
TIME_COL    = "SubmitTime"
REG_Y_COL   = "RemoteWallClockTime"  # 不参与训练

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ============== Utils ==============
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
    # 按类数返回 XGB 的 objective 及附加参数
    if n_classes <= 1:
        raise ValueError("n_classes must be >= 2 for XGBoost training.")
    if n_classes == 2:
        return "binary:logistic", {}
    else:
        return "multi:softprob", {"num_class": n_classes}

def _fit_with_optuna(Xtr, ytr, Xva, yva, n_classes: int, class_weights=None):
    # 使用 Optuna 搜参，最大化宏 F1（最小化负 F1）
    objective, extra = _xgb_objective_and_extras(n_classes)

    def obj(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_estimators": 2000,
            "random_state": RANDOM_SEED,
            "objective": objective,
            "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
            "verbosity": 0,
            # 纯 CPU 时可考虑打开更快实现：
            # "tree_method": "hist",
            **extra,
        }

        model = XGBClassifier(**params)
        model.set_params(early_stopping_rounds=100)

        fit_kwargs = {"eval_set": [(Xva, yva)], "verbose": False}
        if class_weights is not None:
            # 将 class_weights -> sample_weight
            wmap = {int(c): float(w) for c, w in zip(sorted(np.unique(ytr)), class_weights)}
            sw_tr = np.asarray([wmap[int(c)] for c in ytr], dtype=float)
            fit_kwargs["sample_weight"] = sw_tr

        model.fit(Xtr, ytr, **fit_kwargs)
        pred = model.predict(Xva)
        return -f1_score(yva, pred, average="macro", zero_division=0)

    study = optuna.create_study(direction="minimize")
    study.optimize(obj, n_trials=N_TRIALS)
    best = study.best_params
    best.update({
        "n_estimators": 2000,
        "random_state": RANDOM_SEED,
        "objective": objective,
        "eval_metric": "mlogloss" if n_classes > 2 else "logloss",
        "verbosity": 0,
        # "tree_method": "hist",  # 同上
        **extra,
    })
    return best

def _atomic_joblib_save(model, path: str):
    tmp = path + ".tmp"
    dump(model, tmp)
    os.replace(tmp, path)

def _atomic_csv_save(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _encode_labels(ytr: np.ndarray, yva: np.ndarray):
    """
    将（可能不连续、不从0开始的）标签，统一编码为 0..K-1。
    返回：ytr_enc, yva_enc, classes_sorted(原始标签升序), label2idx(原->编), idx2label(编->原)
    """
    classes_sorted = np.unique(np.concatenate([ytr, yva]).astype(int))
    label2idx: Dict[int, int] = {int(c): i for i, c in enumerate(classes_sorted)}
    idx2label: Dict[int, int] = {i: int(c) for i, c in enumerate(classes_sorted)}
    ytr_enc = np.array([label2idx[int(c)] for c in ytr], dtype=int)
    yva_enc = np.array([label2idx[int(c)] for c in yva], dtype=int)
    return ytr_enc, yva_enc, classes_sorted, label2idx, idx2label

# ============== Load TopN ==============
with open(TOPN_JSON, "r", encoding="utf-8") as f:
    top_programids = json.load(f)

# 若指定了单 PID，则只跑这一个
if SINGLE_PID is not None:
    top_programids = [SINGLE_PID]

summary_rows = []

# ============== Train per PID ==============
for pid in top_programids:
    train_path = os.path.join(SPLIT_DIR, f"train_top{pid}.csv")
    val_path   = os.path.join(SPLIT_DIR, f"val_top{pid}.csv")

    # 已完成（模型+报告都在）就跳过 —— 支持断点续跑
    mpath_optuna  = os.path.join(MODEL_DIR, f"xgb_cls_pid{pid}_optuna.joblib")
    mpath_const   = os.path.join(MODEL_DIR, f"xgb_cls_pid{pid}_constant.joblib")
    rpt_path      = os.path.join(REPORT_DIR, f"pid_{pid}_classification_report.csv")
    if (os.path.exists(mpath_optuna) or os.path.exists(mpath_const)) and os.path.exists(rpt_path):
        print(f"[SKIP] PID {pid}: model & report already exist.")
        continue

    # 轻量“锁”：避免同一 PID 被重复训练
    lock_dir = os.path.join(MODEL_DIR, f".lock_pid_{pid}")
    try:
        os.mkdir(lock_dir)  # 获锁（如果已存在会抛异常）
    except FileExistsError:
        print(f"[LOCK] PID {pid} is running elsewhere. Skip.")
        continue

    try:
        if not os.path.exists(train_path):
            print(f"[SKIP] PID {pid}: train file not found")
            continue

        df_tr = pd.read_csv(train_path).dropna(subset=[TARGET_COL])
        df_va = pd.read_csv(val_path).dropna(subset=[TARGET_COL]) if os.path.exists(val_path) else pd.DataFrame(columns=df_tr.columns)

        feat_cols = _numeric_features(df_tr, drop_cols=(TARGET_COL, TIME_COL, REG_Y_COL))
        if not feat_cols:
            print(f"[WARN] PID {pid}: no numeric features left.")
            continue
        _maybe_save_features(feat_cols)

        if len(df_va) == 0:
            print(f"[WARN] PID {pid}: no validation split, skip.")
            continue

        Xtr, ytr = df_tr[feat_cols].values, df_tr[TARGET_COL].astype(int).values
        Xva, yva = df_va[feat_cols].values, df_va[TARGET_COL].astype(int).values

        classes = np.unique(ytr)
        n_classes = len(classes)

        if n_classes == 1:
            # 单类：用恒预测器
            const_label = int(classes[0])
            final = DummyClassifier(strategy="constant", constant=const_label)
            final.fit(Xtr, ytr)
            pred = final.predict(Xva)
            acc  = accuracy_score(yva, pred)
            f1m  = f1_score(yva, pred, average="macro", zero_division=0)

            mpath = os.path.join(MODEL_DIR, f"xgb_cls_pid{pid}_constant.joblib")
            _atomic_joblib_save(final, mpath)

            rpt = classification_report(yva, pred, output_dict=True, zero_division=0)
            _atomic_csv_save(pd.DataFrame(rpt), os.path.join(REPORT_DIR, f"pid_{pid}_classification_report.csv"))

            summary_rows.append({
                "ProgramID_encoded": pid,
                "TrainSize": len(df_tr),
                "ValSize": len(df_va),
                "Accuracy": acc,
                "F1_macro": f1m,
                "BestParams": json.dumps({"model": "constant", "label": const_label}),
                "ModelPath": mpath
            })
            print(f"PID {pid} | SINGLE-CLASS -> const={const_label} | Acc={acc:.3f} | F1_macro={f1m:.3f}")

        else:
            # === 关键：统一标签编码到 0..K-1 ===
            ytr_enc, yva_enc, classes_sorted, label2idx, idx2label = _encode_labels(ytr, yva)
            n_classes = len(classes_sorted)

            # class_weight 用编码后的标签
            cls_weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr_enc)
            best = _fit_with_optuna(Xtr, ytr_enc, Xva, yva_enc, n_classes=n_classes, class_weights=cls_weights)

            final = XGBClassifier(**best).set_params(early_stopping_rounds=100)

            # sample_weight 映射：编码空间 0..K-1
            wmap = {int(c): float(w) for c, w in zip(range(n_classes), cls_weights)}
            sw_tr = np.asarray([wmap[int(c)] for c in ytr_enc], dtype=float)

            final.fit(Xtr, ytr_enc, sample_weight=sw_tr, eval_set=[(Xva, yva_enc)], verbose=False)

            # 预测与评估（编码空间）
            pred_enc = final.predict(Xva)
            acc  = accuracy_score(yva_enc, pred_enc)
            f1m  = f1_score(yva_enc, pred_enc, average="macro", zero_division=0)

            # 报告里反编码回原始标签，便于阅读
            yva_raw  = np.array([idx2label[int(i)] for i in yva_enc], dtype=int)
            pred_raw = np.array([idx2label[int(i)] for i in pred_enc], dtype=int)
            rpt = classification_report(yva_raw, pred_raw, output_dict=True, zero_division=0)

            mpath = os.path.join(MODEL_DIR, f"xgb_cls_pid{pid}_optuna.joblib")
            _atomic_joblib_save(final, mpath)

            _atomic_csv_save(pd.DataFrame(rpt), os.path.join(REPORT_DIR, f"pid_{pid}_classification_report.csv"))

            best_to_save = dict(best)
            best_to_save["label_mapping"] = {int(k): int(v) for k, v in label2idx.items()}

            summary_rows.append({
                "ProgramID_encoded": pid,
                "TrainSize": len(df_tr),
                "ValSize": len(df_va),
                "Accuracy": acc,
                "F1_macro": f1m,
                "BestParams": json.dumps(best_to_save),
                "ModelPath": mpath
            })
            print(f"PID {pid} | Acc={acc:.3f} | F1_macro={f1m:.3f}")

    finally:
        try:
            os.rmdir(lock_dir)
        except Exception:
            pass

# ===== Others 与 汇总：只在“整批模式”执行（避免并发冲突） =====
if SINGLE_PID is None:
    print("\n>> Training fallback XGBClassifier for Others ...")
    df_all_tr = pd.read_csv(ALL_TRAIN)
    df_all_va = pd.read_csv(ALL_VAL)
    with open(TOPN_JSON, "r", encoding="utf-8") as f:
        top_set = set(json.load(f))

    df_ot_tr = df_all_tr[~df_all_tr["ProgramID_encoded"].isin(top_set)].dropna(subset=[TARGET_COL])
    df_ot_va = df_all_va[~df_all_va["ProgramID_encoded"].isin(top_set)].dropna(subset=[TARGET_COL])

    if len(df_ot_tr) and len(df_ot_va):
        feat_cols = _numeric_features(df_ot_tr, drop_cols=(TARGET_COL, TIME_COL, REG_Y_COL))
        _maybe_save_features(feat_cols)

        if not feat_cols:
            print("[SKIP] Others: no numeric features left.")
        else:
            Xtr, ytr = df_ot_tr[feat_cols].values, df_ot_tr[TARGET_COL].astype(int).values
            Xva, yva = df_ot_va[feat_cols].values, df_ot_va[TARGET_COL].astype(int).values

            classes = np.unique(ytr)
            n_classes = len(classes)

            if n_classes == 1:
                const_label = int(classes[0])
                final = DummyClassifier(strategy="constant", constant=const_label)
                final.fit(Xtr, ytr)
                pred = final.predict(Xva)
                acc  = accuracy_score(yva, pred)
                f1m  = f1_score(yva, pred, average="macro", zero_division=0)

                mpath = os.path.join(MODEL_DIR, "xgb_cls_others_constant.joblib")
                _atomic_joblib_save(final, mpath)

                rpt = classification_report(yva, pred, output_dict=True, zero_division=0)
                _atomic_csv_save(pd.DataFrame(rpt), os.path.join(REPORT_DIR, "others_classification_report.csv"))

                summary_rows.append({
                    "ProgramID_encoded": "Others",
                    "TrainSize": len(df_ot_tr),
                    "ValSize": len(df_ot_va),
                    "Accuracy": acc,
                    "F1_macro": f1m,
                    "BestParams": json.dumps({"model": "constant", "label": const_label}),
                    "ModelPath": mpath
                })
                print(f"Others | SINGLE-CLASS -> const={const_label} | Acc={acc:.3f} | F1_macro={f1m:.3f}")

            else:
                # === Others 也做统一编码 ===
                ytr_enc, yva_enc, classes_sorted, label2idx, idx2label = _encode_labels(ytr, yva)
                n_classes = len(classes_sorted)

                cls_weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=ytr_enc)
                best = _fit_with_optuna(Xtr, ytr_enc, Xva, yva_enc, n_classes=n_classes, class_weights=cls_weights)

                final = XGBClassifier(**best).set_params(early_stopping_rounds=100)
                wmap = {int(c): float(w) for c, w in zip(range(n_classes), cls_weights)}
                sw_tr = np.asarray([wmap[int(c)] for c in ytr_enc], dtype=float)

                final.fit(Xtr, ytr_enc, sample_weight=sw_tr, eval_set=[(Xva, yva_enc)], verbose=False)

                pred_enc = final.predict(Xva)
                acc  = accuracy_score(yva_enc, pred_enc)
                f1m  = f1_score(yva_enc, pred_enc, average="macro", zero_division=0)

                yva_raw  = np.array([idx2label[int(i)] for i in yva_enc], dtype=int)
                pred_raw = np.array([idx2label[int(i)] for i in pred_enc], dtype=int)
                rpt = classification_report(yva_raw, pred_raw, output_dict=True, zero_division=0)

                mpath = os.path.join(MODEL_DIR, "xgb_cls_others_optuna.joblib")
                _atomic_joblib_save(final, mpath)

                _atomic_csv_save(pd.DataFrame(rpt), os.path.join(REPORT_DIR, "others_classification_report.csv"))

                best_to_save = dict(best)
                best_to_save["label_mapping"] = {int(k): int(v) for k, v in label2idx.items()}

                summary_rows.append({
                    "ProgramID_encoded": "Others",
                    "TrainSize": len(df_ot_tr),
                    "ValSize": len(df_ot_va),
                    "Accuracy": acc,
                    "F1_macro": f1m,
                    "BestParams": json.dumps(best_to_save),
                    "ModelPath": mpath
                })
    else:
        print("[SKIP] Others: insufficient data.")

    # ===== 汇总表（整批模式才写） =====
    sum_path = os.path.join(REPORT_DIR, "v5.4_cls_xgb_summary.csv")
    new_df = pd.DataFrame(summary_rows)
    if os.path.exists(sum_path):
        old_df = pd.read_csv(sum_path)
        out_df = pd.concat([old_df, new_df], ignore_index=True)
        out_df = out_df.drop_duplicates(subset=["ProgramID_encoded"], keep="last")
    else:
        out_df = new_df
    _atomic_csv_save(out_df, sum_path)
    print("\nAll XGB classification models finished. Summary ->", sum_path)
