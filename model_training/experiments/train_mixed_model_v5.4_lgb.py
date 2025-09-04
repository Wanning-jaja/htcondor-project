# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, warnings, sys
from typing import Tuple, List

import numpy as np
import pandas as pd
import optuna
from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

# ---- 路径与常量 ----
SPLIT_DIR     = "/home/master/wzheng/projects/model_training/data/top40_splits"
TOPN_JSON     = "/home/master/wzheng/projects/model_training/data/top40_programid_list.json"
MODEL_DIR     = "/home/master/wzheng/projects/model_training/models/v5.4_cls_lgb"
REPORT_DIR    = "/home/master/wzheng/projects/model_training/reports/v5.4_cls_lgb"
FEATURES_JSON = "/home/master/wzheng/projects/model_training/models/v5.4_cls_features.json"

ALL_TRAIN = "/home/master/wzheng/projects/model_training/data/40train.csv"
ALL_VAL   = "/home/master/wzheng/projects/model_training/data/40val.csv"

RANDOM_SEED = 42
N_TRIALS    = 50

TARGET_COL = "BucketLabel"
TIME_COL   = "SubmitTime"
REG_Y_COL  = "RemoteWallClockTime"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---- 工具函数 ----
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

def _fit_with_optuna(Xtr: pd.DataFrame, ytr: np.ndarray,
                     Xva: pd.DataFrame, yva: np.ndarray,
                     n_classes: int, class_weights=None):
    # 按类别数动态设置 LightGBM 的 objective/num_class，并用 Optuna 搜参。
    if n_classes <= 1:
        raise ValueError("n_classes must be >= 2 for training.")
    obj = "binary" if n_classes == 2 else "multiclass"

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", -1, 12),
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "n_estimators": 3000,
            "random_state": RANDOM_SEED,
            "objective": obj,
        }
        if obj == "multiclass":
            params["num_class"] = n_classes

        model = LGBMClassifier(**params)
        fit_kwargs = {
            "eval_set": [(Xva, yva)],
            "callbacks": [early_stopping(100), log_evaluation(0)]
        }

        if class_weights is not None:
            cw = {int(c): float(w) for c, w in zip(sorted(np.unique(ytr)), class_weights)}
            model.set_params(class_weight=cw)

        model.fit(Xtr, ytr, **fit_kwargs)
        pred = model.predict(Xva, num_iteration=getattr(model, "best_iteration_", None))
        return -f1_score(yva, pred, average="macro", zero_division=0)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_params
    best.update({"n_estimators": 3000, "random_state": RANDOM_SEED, "objective": obj})
    if obj == "multiclass":
        best["num_class"] = n_classes
    return best

# ---- 读取 TopN ProgramIDs ----
with open(TOPN_JSON, 'r', encoding="utf-8") as f:
    top_programids = json.load(f)

summary_rows = []

# ---- 按 PID 训练 ----
for pid in top_programids:
    tr_path = os.path.join(SPLIT_DIR, f"train_top{pid}.csv")
    va_path = os.path.join(SPLIT_DIR, f"val_top{pid}.csv")
    if not os.path.exists(tr_path):
        print(f"[SKIP] PID {pid}: train file not found")
        continue

    df_tr = pd.read_csv(tr_path).dropna(subset=[TARGET_COL])
    df_va = pd.read_csv(va_path).dropna(subset=[TARGET_COL]) if os.path.exists(va_path) else pd.DataFrame(columns=df_tr.columns)

    feat_cols = _numeric_features(df_tr, drop_cols=(TARGET_COL, TIME_COL, REG_Y_COL))
    if not feat_cols:
        print(f"[WARN] PID {pid}: no numeric features left.")
        continue
    _maybe_save_features(feat_cols)

    if len(df_va) == 0:
        print(f"[WARN] PID {pid}: no validation split, skip.")
        continue

    # ? 全程 DataFrame（带列名）
    Xtr = df_tr[feat_cols]
    ytr = df_tr[TARGET_COL].astype(int).values
    Xva = df_va[feat_cols]
    yva = df_va[TARGET_COL].astype(int).values

    classes = np.unique(ytr)
    n_classes = len(classes)

    if n_classes == 1:
        # 单类：不训练 LightGBM，使用恒预测器
        const_label = int(classes[0])
        final = DummyClassifier(strategy='constant', constant=const_label)
        final.fit(Xtr, ytr)  # 统一接口
        pred = final.predict(Xva)
        acc  = accuracy_score(yva, pred)
        f1m  = f1_score(yva, pred, average="macro", zero_division=0)

        mpath = os.path.join(MODEL_DIR, f"lgb_cls_pid{pid}_constant.joblib")
        dump(final, mpath)

        rpt = classification_report(yva, pred, output_dict=True, zero_division=0)
        pd.DataFrame(rpt).to_csv(os.path.join(REPORT_DIR, f"pid_{pid}_classification_report.csv"))

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
        # 二分类/多分类：Optuna + LightGBM
        cls_weights = compute_class_weight('balanced', classes=classes, y=ytr)
        best = _fit_with_optuna(Xtr, ytr, Xva, yva, n_classes=n_classes, class_weights=cls_weights)

        final = LGBMClassifier(**best)
        cw = {int(c): float(w) for c, w in zip(classes, cls_weights)}
        final.set_params(class_weight=cw)
        final.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[early_stopping(100), log_evaluation(0)])

        pred = final.predict(Xva, num_iteration=getattr(final, "best_iteration_", None))
        acc  = accuracy_score(yva, pred)
        f1m  = f1_score(yva, pred, average="macro", zero_division=0)

        mpath = os.path.join(MODEL_DIR, f"lgb_cls_pid{pid}_optuna.joblib")
        dump(final, mpath)

        rpt = classification_report(yva, pred, output_dict=True, zero_division=0)
        pd.DataFrame(rpt).to_csv(os.path.join(REPORT_DIR, f"pid_{pid}_classification_report.csv"))

        summary_rows.append({
            "ProgramID_encoded": pid,
            "TrainSize": len(df_tr),
            "ValSize": len(df_va),
            "Accuracy": acc,
            "F1_macro": f1m,
            "BestParams": json.dumps(best),
            "ModelPath": mpath
        })
        print(f"PID {pid} | Acc={acc:.3f} | F1_macro={f1m:.3f}")

# ---- Others 兜底 ----
print("\n>> Training fallback for Others ...")
df_all_tr = pd.read_csv(ALL_TRAIN)
df_all_va = pd.read_csv(ALL_VAL)
with open(TOPN_JSON, 'r', encoding="utf-8") as f:
    top_set = set(json.load(f))

df_ot_tr = df_all_tr[~df_all_tr["ProgramID_encoded"].isin(top_set)].dropna(subset=[TARGET_COL])
df_ot_va = df_all_va[~df_all_va["ProgramID_encoded"].isin(top_set)].dropna(subset=[TARGET_COL])

if len(df_ot_tr) and len(df_ot_va):
    feat_cols = _numeric_features(df_ot_tr, drop_cols=(TARGET_COL, TIME_COL, REG_Y_COL))
    _maybe_save_features(feat_cols)

    if not feat_cols:
        print("[SKIP] Others: no numeric features left.")
    else:
        # ? 这里也全程 DataFrame
        Xtr = df_ot_tr[feat_cols]
        ytr = df_ot_tr[TARGET_COL].astype(int).values
        Xva = df_ot_va[feat_cols]
        yva = df_ot_va[TARGET_COL].astype(int).values

        classes = np.unique(ytr)
        n_classes = len(classes)

        if n_classes == 1:
            const_label = int(classes[0])
            final = DummyClassifier(strategy='constant', constant=const_label)
            final.fit(Xtr, ytr)
            pred = final.predict(Xva)
            acc  = accuracy_score(yva, pred)
            f1m  = f1_score(yva, pred, average="macro", zero_division=0)

            mpath = os.path.join(MODEL_DIR, "lgb_cls_others_constant.joblib")
            dump(final, mpath)

            rpt = classification_report(yva, pred, output_dict=True, zero_division=0)
            pd.DataFrame(rpt).to_csv(os.path.join(REPORT_DIR, "others_classification_report.csv"))

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
            cls_weights = compute_class_weight('balanced', classes=classes, y=ytr)
            best = _fit_with_optuna(Xtr, ytr, Xva, yva, n_classes=n_classes, class_weights=cls_weights)

            final = LGBMClassifier(**best)
            final.set_params(class_weight={int(c): float(w) for c, w in zip(classes, cls_weights)})
            final.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[early_stopping(100), log_evaluation(0)])

            pred = final.predict(Xva, num_iteration=getattr(final, "best_iteration_", None))
            acc  = accuracy_score(yva, pred)
            f1m  = f1_score(yva, pred, average="macro", zero_division=0)

            mpath = os.path.join(MODEL_DIR, "lgb_cls_others_optuna.joblib")
            dump(final, mpath)

            rpt = classification_report(yva, pred, output_dict=True, zero_division=0)
            pd.DataFrame(rpt).to_csv(os.path.join(REPORT_DIR, "others_classification_report.csv"))

            summary_rows.append({
                "ProgramID_encoded": "Others",
                "TrainSize": len(df_ot_tr),
                "ValSize": len(df_ot_va),
                "Accuracy": acc,
                "F1_macro": f1m,
                "BestParams": json.dumps(best),
                "ModelPath": mpath
            })
else:
    print("[SKIP] Others: insufficient data.")

# ---- 汇总 ----
sum_path = os.path.join(REPORT_DIR, "v5.4_cls_lgb_summary.csv")
pd.DataFrame(summary_rows).to_csv(sum_path, index=False)
print("\nAll classification models finished. Summary ->", sum_path)
