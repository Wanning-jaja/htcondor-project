# -*- coding: utf-8 -*-

#4.evaluate_predictions_v5.3.py

#���ܣ�
#- ��ȡԤ���������� v5.3_ensemble_predictions.csv��������ȫ��������
#  1) ������rows��RMSE��MAE��MAPE��Bias��R2��Bucket Accuracy��Accuracy+-1��
#     under/over/hit ռ�ȡ�ƽ��ƫ�루Ͱ�������������д��ۣ��͹�/�߹���ͬȨ�أ���
#  2) �� PID ָ���n��RMSE��MAE��Bucket Accuracy��Accuracy+-1��under/over/hit ռ�ȡ�ƽ��ƫ�롣
#  3) ����ʵͰ��Ԥ��Ͱ�Ļ�������csv + ����ͼ����
#  4) ����ͳ����ƫ����ȣ��룩ͳ�ƣ�csv + ���ӻ�����

#- �����ֶ�ȱʧ��
#  ��û�� true_bucket / pred_bucket�����������ֵ��RemoteWallClockTime / PredictedRemoteWallClockTime�����߽��Զ����㡣


from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ========= ·������ =========
INPUT_CSV  = "/home/master/wzheng/projects/model_training/preds/v5.3_ensemble_predictions_C.csv"  # ���Ԥ�����
OUTPUT_DIR = "/home/master/wzheng/projects/model_training/evaluation/ABCD/C_v5.3_ensemble"           # �������Ŀ¼
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========= �ֶ�Լ�������㵱ǰԤ���ļ�һ�£�=========
Y_TRUE_COL  = "RemoteWallClockTime"
Y_PRED_COL  = "PredictedRemoteWallClockTime"
PID_COL     = "ProgramID_encoded"     # �������ڣ����� ProgramID ����������ʱ����
TRUE_BKT    = "true_bucket"
PRED_BKT    = "pred_bucket"
DIR_COL     = "direction"             # under / over / hit�������ִ�Сд��
DEV_COL     = "deviation"             # Ԥ��Ͱ - ��ʵͰ�������������ɸ���

# ========= Ͱ���ã���ѵ��һ�£�=========
# �߽磺 [0,600), [600,1800), [1800,3600), [3600,7200), [7200,14400),
#        [14400,21600), [21600,28800), [28800,43200), [43200,86400), [86400, +inf)
BUCKET_EDGES = np.array([0, 600, 1800, 3600, 7200, 14400, 21600, 28800, 43200, 86400, np.inf], dtype=float)
BUCKET_IDS   = list(range(len(BUCKET_EDGES)-1))  # 0..9

# ========= �������д��ۣ��͹�����=========
LAM_UNDER = 0.6
LAM_OVER  = 0.2
SCALE     = 1000.0  # ��ѵ��/���ɽ׶ε�����һ��

# ========= ���ߺ��� =========
def bucketize_seconds(arr: np.ndarray) -> np.ndarray:
#    ���� 0..9 ��Ͱ���
    return np.digitize(np.asarray(arr, dtype=float), BUCKET_EDGES[1:-1], right=False)

def safe_mape(y, p) -> float:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    m = np.abs(y) > 1e-9
    return float(np.mean(np.abs((p[m] - y[m]) / y[m])) * 100.0) if np.any(m) else np.nan

def overall_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    tb: np.ndarray, pb: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mape = safe_mape(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan
    acc  = float(np.mean(pb == tb))
    acc1 = float(np.mean(np.abs(pb - tb) <= 1))
    return {"rows": len(y_true), "RMSE": rmse, "MAE": mae, "MAPE(%)": mape,
            "Bias": bias, "R2": r2, "Accuracy": acc, "Accuracy+-1": acc1}

def direction_from_tb_pb(tb_i: int, pb_i: int) -> str:
    if tb_i == pb_i: return "hit"
    return "over" if pb_i > tb_i else "under"

def direction_sensitive_cost(dev: np.ndarray,
                             lam_under=LAM_UNDER, lam_over=LAM_OVER, scale=SCALE) -> float:
    # dev = pred_bucket - true_bucket
    under = np.clip(-dev, 0, None)  # �������͹���Ͱ��
    over  = np.clip( dev, 0, None)  # �������߹���Ͱ��
    return float(np.mean(lam_under * under + lam_over * over) * scale)

# ========= ������ =========
def main():
    df = pd.read_csv(INPUT_CSV)
    cols = df.columns.tolist()
    print(f"[INFO] Loaded {len(df)} rows with {len(cols)} columns.")

    # �� ProgramID_encoded
    if PID_COL not in cols:
        if "ProgramID" in cols:
            df[PID_COL] = pd.Categorical(df["ProgramID"]).codes
            print("[WARN] ProgramID_encoded missing; derived temporary codes from ProgramID.")
        else:
            df[PID_COL] = -1  # ��һ����
            print("[WARN] ProgramID(_encoded) missing; using a single group.")

    # Ͱ����ȱʧ�����
    if TRUE_BKT not in cols and Y_TRUE_COL in cols:
        df[TRUE_BKT] = bucketize_seconds(df[Y_TRUE_COL].values)
        print("[INFO] true_bucket reconstructed from true seconds.")
    if PRED_BKT not in cols and Y_PRED_COL in cols:
        df[PRED_BKT] = bucketize_seconds(df[Y_PRED_COL].values)
        print("[INFO] pred_bucket reconstructed from predicted seconds.")

    # ������ƫ���ȱʧ�����
    if DIR_COL not in cols or DEV_COL not in cols:
        tb = df[TRUE_BKT].to_numpy(dtype=int)
        pb = df[PRED_BKT].to_numpy(dtype=int)
        dev = pb - tb
        df[DEV_COL] = dev
        df[DIR_COL] = np.where(dev == 0, "hit", np.where(dev > 0, "over", "under"))
        print("[INFO] direction / deviation reconstructed from buckets.")

    # ����ȱʧ
    keep = df[[Y_TRUE_COL, Y_PRED_COL, TRUE_BKT, PRED_BKT, PID_COL]].dropna().index
    if len(keep) < len(df):
        print(f"[WARN] Dropped {len(df)-len(keep)} rows with NA in key columns.")
    df = df.loc[keep].reset_index(drop=True)

    # ======== ����ָ�� ========
    y  = df[Y_TRUE_COL].to_numpy(dtype=float)
    yp = df[Y_PRED_COL].to_numpy(dtype=float)
    tb = df[TRUE_BKT].to_numpy(dtype=int)
    pb = df[PRED_BKT].to_numpy(dtype=int)
    overall = overall_metrics(y, yp, tb, pb)

    # ����ռ�� / ƽ��ƫ��
    dir_vals = df[DIR_COL].str.lower().values
    under_rate = float(np.mean(dir_vals == "under"))
    over_rate  = float(np.mean(dir_vals == "over"))
    hit_rate   = float(np.mean(dir_vals == "hit"))
    avg_dev    = float(df[DEV_COL].mean())
    # �������д���
    dir_cost   = direction_sensitive_cost(df[DEV_COL].to_numpy(dtype=float))

    overall_row = {
        **overall,
        "under_rate": under_rate,
        "over_rate":  over_rate,
        "hit_rate":   hit_rate,
        "avg_deviation(bucket)": avg_dev,
        f"dir_cost(l_under={LAM_UNDER},l_over={LAM_OVER},scale={SCALE})": dir_cost
    }
    overall_df = pd.DataFrame([overall_row])
    overall_path = os.path.join(OUTPUT_DIR, "overall_metrics.csv")
    overall_df.to_csv(overall_path, index=False)
    print(f"[OK] overall metrics -> {overall_path}")

    # ======== �� PID ָ�� ========
    def _rmse_pid(g):
        yt = g[Y_TRUE_COL].to_numpy(dtype=float); yp = g[Y_PRED_COL].to_numpy(dtype=float)
        return float(np.sqrt(np.mean((yp-yt)**2)))
    def _mae_pid(g):
        yt = g[Y_TRUE_COL].to_numpy(dtype=float); yp = g[Y_PRED_COL].to_numpy(dtype=float)
        return float(np.mean(np.abs(yp-yt)))
    def _acc_pid(g):
        return float(np.mean(g[PRED_BKT].to_numpy(dtype=int) == g[TRUE_BKT].to_numpy(dtype=int)))
    def _acc1_pid(g):
        d = g[PRED_BKT].to_numpy(dtype=int) - g[TRUE_BKT].to_numpy(dtype=int)
        return float(np.mean(np.abs(d) <= 1))
    def _dir_cost_pid(g):
        return direction_sensitive_cost(g[DEV_COL].to_numpy(dtype=float))

    pid_grp = df.groupby(PID_COL)
    per_pid = pid_grp.apply(lambda g: pd.Series({
        "n": len(g),
        "RMSE": _rmse_pid(g),
        "MAE":  _mae_pid(g),
        "Accuracy": _acc_pid(g),
        "Accuracy+-1": _acc1_pid(g),
        "under_rate": float(np.mean(g[DIR_COL].str.lower().values=="under")),
        "over_rate":  float(np.mean(g[DIR_COL].str.lower().values=="over")),
        "hit_rate":   float(np.mean(g[DIR_COL].str.lower().values=="hit")),
        "avg_deviation(bucket)": float(g[DEV_COL].mean()),
        f"dir_cost(l_under={LAM_UNDER},l_over={LAM_OVER},scale={SCALE})": _dir_cost_pid(g)
    })).reset_index()
    per_pid_path = os.path.join(OUTPUT_DIR, "per_pid_metrics.csv")
    per_pid.to_csv(per_pid_path, index=False)
    print(f"[OK] per-PID metrics -> {per_pid_path}")

    # ======== ����������ʵͰ��Ԥ��Ͱ�� ========
    cm = pd.crosstab(df[TRUE_BKT], df[PRED_BKT]).reindex(index=BUCKET_IDS, columns=BUCKET_IDS, fill_value=0)
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
    cm.to_csv(cm_path)
    print(f"[OK] confusion matrix -> {cm_path}")

    # ����ͼ
    plt.figure(figsize=(8,6))
    plt.imshow(cm.values, aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(ticks=np.arange(len(BUCKET_IDS)), labels=BUCKET_IDS)
    plt.yticks(ticks=np.arange(len(BUCKET_IDS)), labels=BUCKET_IDS)
    plt.xlabel("pred_bucket"); plt.ylabel("true_bucket")
    plt.title("Confusion Matrix (bucket)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_heatmap.png"))
    plt.close()

    # ======== ���������ƫ��룩ͳ�� ========
    # ƫ���룺���ļ���û�� deviation_seconds���ɰ���ɽű��Ķ��岹�㣨������ԭ���߼���
    if "deviation_seconds" not in df.columns:
        # ���½��ֵ�
        upper_bounds = dict(zip(BUCKET_IDS, BUCKET_EDGES[1:]))
        lower_bounds = dict(zip(BUCKET_IDS, BUCKET_EDGES[:-1]))
        def _dev_seconds(row):
            tb_i, pb_i = int(row[TRUE_BKT]), int(row[PRED_BKT])
            if tb_i == pb_i: return 0.0
            if pb_i > tb_i:  # Over��Ԥ�����
                return float(row[Y_PRED_COL] - upper_bounds[tb_i])
            else:            # Under��Ԥ�����
                return float(lower_bounds[tb_i] - row[Y_PRED_COL])
        df["deviation_seconds"] = df.apply(_dev_seconds, axis=1)

    dir_counts = df[DIR_COL].str.capitalize().value_counts().reindex(["Hit","Under","Over"]).fillna(0).astype(int)
    dir_counts.to_csv(os.path.join(OUTPUT_DIR, "direction_counts.csv"))
    # ����ͼ
    plt.figure()
    ax = dir_counts.plot(kind="bar", color=["green","red","orange"])
    for i, v in enumerate(dir_counts.values):
        ax.text(i, v + max(dir_counts.values)*0.01, str(int(v)), ha="center", va="bottom")
    plt.ylabel("Count"); plt.title("Bucket Prediction Result (Hit / Under / Over)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "direction_counts_bar.png"))
    plt.close()

    # ƫ���������ͼ��������������
    err = df[df[DIR_COL].str.lower() != "hit"]
    if len(err) > 0:
        plt.figure()
        err.boxplot(column="deviation_seconds", by=DIR_COL)
        plt.title("Deviation by Error Direction (seconds)")
        plt.suptitle("")
        plt.ylabel("Deviation (seconds)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "deviation_seconds_boxplot.png"))
        plt.close()

    print(f"\nAll evaluation artifacts saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
