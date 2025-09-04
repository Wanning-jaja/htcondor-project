# -*- coding: utf-8 -*-

#2.2.train_stacking_v5.3.py — direction-sensitive + safety margin

from __future__ import annotations
import os, json, warnings
from typing import Dict
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=UserWarning)

# ====== PATHS ======
LGB_REPORT_DIR = "/home/master/wzheng/projects/model_training/reports/v5.3_lgb"
XGB_REPORT_DIR = "/home/master/wzheng/projects/model_training/reports/v5.3_xgb"
LGB_OOF = os.path.join(LGB_REPORT_DIR, "oof_preds_v5.3.csv")
XGB_OOF = os.path.join(XGB_REPORT_DIR, "oof_preds_v5.3.csv")

ENSEMBLE_MODEL_DIR  = "/home/master/wzheng/projects/model_training/models/v5.3_ensemble"
ENSEMBLE_REPORT_DIR = "/home/master/wzheng/projects/model_training/reports/v5.3_ensemble"
os.makedirs(ENSEMBLE_MODEL_DIR, exist_ok=True)
os.makedirs(ENSEMBLE_REPORT_DIR, exist_ok=True)

LGB_MODEL_DIR = "/home/master/wzheng/projects/model_training/models/v5.3_lgb"
XGB_MODEL_DIR = "/home/master/wzheng/projects/model_training/models/v5.3_xgb"

# ====== BUCKETS & COST ======
_BUCKET_EDGES = [600, 1800, 3600, 7200, 14400, 21600, 28800, 43200, 86400]
# 方向敏感权重：按你的业务设定“低估更贵”
LAM_UNDER = 0.6  # 低估（预测短于真实）每跨一桶的成本权重
LAM_OVER  = 0.2  # 高估（预测长于真实）每跨一桶的成本权重
SCALE     = 1000.0  # 与 v5.3 训练一致的量纲

W_GRID = np.linspace(0.0, 1.0, 21)  # 权重网格：0,0.05,...,1

def _bucketize(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.digitize(a, _BUCKET_EDGES, right=False)  # 0..9

def ds_penalized_rmse(y_true: np.ndarray, y_pred: np.ndarray,
                      lam_under=LAM_UNDER, lam_over=LAM_OVER, scale=SCALE) -> float:
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
    tb, pb = _bucketize(y_true), _bucketize(y_pred)
    diff  = pb - tb
    # 低估: diff<0（预测桶比真实小）；高估: diff>0
    under = np.clip(-diff, 0, None)  # 正值 = 低估跨了几桶
    over  = np.clip(diff,  0, None)  # 正值 = 高估跨了几桶
    penalty = float(np.mean(lam_under * under + lam_over * over) * scale)
    return rmse + penalty

def overall_metrics(y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y_true = np.asarray(y_true, dtype=float); y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mask = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs((y_pred[mask]-y_true[mask]) / y_true[mask]))*100.0) if np.any(mask) else np.nan
    bias = float(np.mean(y_pred - y_true))
    try: r2 = float(r2_score(y_true, y_pred))
    except: r2 = np.nan
    tb, pb = _bucketize(y_true), _bucketize(y_pred)
    acc = float(np.mean(pb==tb)) if len(tb) else np.nan
    acc1= float(np.mean(np.abs(pb-tb)<=1)) if len(tb) else np.nan
    return {"RMSE":rmse,"MAE":mae,"MAPE(%)":mape,"Bias":bias,"R2":r2,"Accuracy":acc,"Accuracy+-1":acc1}

def _add_seq(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df["__seq__"] = df.groupby("ProgramID_encoded").cumcount(); return df

def _load_oof(path: str) -> pd.DataFrame:
    if not os.path.exists(path): raise FileNotFoundError(path)
    df = pd.read_csv(path)
    need = ["ProgramID_encoded","y_true","y_pred","model","Status"]
    for c in need:
        if c not in df.columns: raise ValueError(f"Missing {c} in {path}")
    return _add_seq(df)

def main():
    print(">>> Loading OOF ...")
    lgb = _load_oof(LGB_OOF); xgb = _load_oof(XGB_OOF)
    lgb = lgb.rename(columns={"y_pred":"y_pred_lgb","Status":"Status_lgb"})
    xgb = xgb.rename(columns={"y_pred":"y_pred_xgb","Status":"Status_xgb"})
    df  = pd.merge(lgb[["ProgramID_encoded","y_true","y_pred_lgb","__seq__","Status_lgb"]],
                   xgb[["ProgramID_encoded","y_true","y_pred_xgb","__seq__","Status_xgb"]],
                   on=["ProgramID_encoded","__seq__"])

    # 如果两侧 y_true 有细微数值差异，以 lgb 为准
    df["y_true"] = df["y_true_xgb"].where(df["y_true_lgb"].isna(), df["y_true_lgb"])
    df = df.drop(columns=[c for c in df.columns if c.startswith("y_true_")])

    rows_cfg, rows_eval = [], []
    pids = sorted(df["ProgramID_encoded"].unique().tolist())

    for pid in pids:
        sub = df[df["ProgramID_encoded"]==pid].copy()
        y  = sub["y_true"].to_numpy()
        px = sub["y_pred_xgb"].to_numpy()
        pl = sub["y_pred_lgb"].to_numpy()

        # 单模得分（方向敏感）
        s_x = ds_penalized_rmse(y, px)
        s_l = ds_penalized_rmse(y, pl)

        # 加权网格
        best_w, best_s = None, np.inf
        for w in W_GRID:
            yh = w*px + (1.0-w)*pl
            s  = ds_penalized_rmse(y, yh)
            if s < best_s: best_s, best_w = s, float(w)

        # 选择：weighted vs xgb_only vs lgb_only
        choices = [("weighted", best_s, best_w, 1.0-best_w),
                   ("xgb_only", s_x, 1.0, 0.0),
                   ("lgb_only", s_l, 0.0, 1.0)]
        choices.sort(key=lambda t: t[1])
        strategy, final_s, w_x, w_l = choices[0]

        # 最终预测 & 指标
        yhat = (w_x*px + w_l*pl) if strategy=="weighted" else (px if strategy=="xgb_only" else pl)
        met  = overall_metrics(y, yhat)

        # —— 方向敏感的“安全裕量”学习：低估部分的 90 分位（秒）——
        resid = yhat - y  # >0: 高估；<0: 低估
        under = resid[resid < 0]  # 低估（负数）
        # 用“绝对低估秒数”的 P90 作为安全裕量（可按需换成 P80/P95）
        safety_shift = float(np.percentile(-under, 90)) if under.size>0 else 0.0

        # 写配置（供预测端使用）
        rows_cfg.append({
            "ProgramID_encoded": pid,
            "strategy": strategy,
            "w_xgb": round(w_x,4),
            "w_lgb": round(w_l,4),
            "safety_shift_sec": round(safety_shift, 1),  # << 方向敏感：低估代价的缓冲
            "lgb_model_path": os.path.join(LGB_MODEL_DIR, f"lgb_model_pid{pid}_optuna.joblib"),
            "xgb_model_path": os.path.join(XGB_MODEL_DIR, f"xgb_model_pid{pid}_optuna.joblib"),
            "lam_under": LAM_UNDER,
            "lam_over": LAM_OVER,
            "scale": SCALE
        })

        rows_eval.append({
            "ProgramID_encoded": pid,
            "strategy": strategy,
            "w_xgb": round(w_x,4),
            "w_lgb": round(w_l,4),
            "score_dir_sensitive": round(final_s,6),
            **{k:met[k] for k in ["RMSE","MAE","MAPE(%)","Bias","R2","Accuracy","Accuracy+-1"]},
            "n_samples": len(sub),
            "Status_lgb": sub["Status_lgb"].mode().iloc[0] if not sub["Status_lgb"].empty else "",
            "Status_xgb": sub["Status_xgb"].mode().iloc[0] if not sub["Status_xgb"].empty else "",
            "safety_shift_sec": round(safety_shift,1)
        })

    cfg_df  = pd.DataFrame(rows_cfg)
    eval_df = pd.DataFrame(rows_eval)

    cfg_path  = os.path.join(ENSEMBLE_MODEL_DIR,  "pid_ensemble_v5.3.csv")
    eval_path = os.path.join(ENSEMBLE_REPORT_DIR, "v5.3_ensemble_evaluation_summary.csv")
    cfg_df.to_csv(cfg_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    print(f">>> Ensemble config saved:   {cfg_path}")
    print(f">>> Ensemble evaluation:     {eval_path}")

    # 摘要
    metrics_cols = ["RMSE","MAE","MAPE(%)","Bias","R2","Accuracy","Accuracy+-1"]
    avg = eval_df[metrics_cols].mean(numeric_only=True)
    print("\n=== Direction-sensitive OOF Averages ===")
    for k in metrics_cols:
        print(f"{k}: {avg[k]:.6f}")

if __name__ == "__main__":
    main()
