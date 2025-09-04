# train_single_pid.py
import sys
import os
import json
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from joblib import dump

# === 配置（注意与主目录一致）===
SPLIT_DIR = "/home/master/wzheng/projects/model_training/data/topN_splits"
MODEL_DIR = "/home/master/wzheng/projects/model_training/models/v3"
REPORT_DIR = "/home/master/wzheng/projects/model_training/reports/v3"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# === 读取命令行参数 ===
pid = sys.argv[1]

train_path = os.path.join(SPLIT_DIR, f"train_top{pid}.csv")
val_path = os.path.join(SPLIT_DIR, f"val_top{pid}.csv")

if not os.path.exists(train_path) or not os.path.exists(val_path):
    print(f"train or val csv not found for ProgramID {pid}, skip.")
    sys.exit(0)

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)
feature_cols = [col for col in df_train.columns if col not in ['RemoteWallClockTime', 'SubmitTime']]
target_col = 'RemoteWallClockTime'

def objective(trial):
    param = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        "n_estimators": 1000,
        "random_state": 42,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0
    }
    model = XGBRegressor(**param)
    model.fit(
        df_train[feature_cols],
        df_train[target_col],
        eval_set=[(df_val[feature_cols], df_val[target_col])],
        early_stopping_rounds=10,
        verbose=False
    )
    pred = model.predict(df_val[feature_cols])
    rmse = mean_squared_error(df_val[target_col], pred, squared=False)
    return rmse

# 启动 Optuna 自动调参
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params.update({
    "n_estimators": 1000,
    "random_state": 42,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0
})

# 重新训练最优模型
model = XGBRegressor(**best_params)
model.fit(
    df_train[feature_cols],
    df_train[target_col],
    eval_set=[(df_val[feature_cols], df_val[target_col])],
    early_stopping_rounds=10,
    verbose=False
)
val_pred = model.predict(df_val[feature_cols])
rmse = mean_squared_error(df_val[target_col], val_pred, squared=False)
mae = mean_absolute_error(df_val[target_col], val_pred)

# 保存模型
model_path = os.path.join(MODEL_DIR, f"xgb_model_pid{pid}_optuna.joblib")
dump(model, model_path)

# 保存单独的结果
result = {
    "ProgramID_encoded": pid,
    "TrainSize": len(df_train),
    "ValSize": len(df_val),
    "RMSE": rmse,
    "MAE": mae,
    "BestParams": json.dumps(best_params)
}
with open(os.path.join(REPORT_DIR, f"result_pid{pid}.json"), "w") as f:
    json.dump(result, f, indent=2)

print(f"PID {pid}: RMSE={rmse:.4f}, MAE={mae:.4f}, Params={best_params}")
