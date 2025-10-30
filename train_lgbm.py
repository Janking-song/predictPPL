# train_lgbm.py
import os
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib


# 加载数据集
def load_data(path):
    df = pd.read_csv(path)
    # print(df.shape)  # (900, 254)
    return df


def prepare_features(df):
    
    # 寻找lightGBM预测的目标列
    if 'log_perplexity' in df.columns:
        target_col = 'log_perplexity'
    else:
        print("Not found log_perplexity")

    # 把 df 拆成 X（特征）和 y（目标），并去掉明显不该作为特征的列
    cols_to_remove = ['decouple_list', 'perplexity', target_col]
    
    cols_to_remove = [c for c in cols_to_remove if c in df.columns and c != 'model_name']

    X = df.drop(columns=cols_to_remove, errors='ignore')
    y = df[target_col].astype(float)

    # 处理无法用于模型训练的object类型特征数据
    for c in X.columns:
        if X[c].dtype == object:
            le = LabelEncoder()
            X[c] = le.fit_transform(X[c].astype(str))

    # 处理训练集中的INF数据和检查并处理缺失值，缺失值不需要填充
    X = X.replace([np.inf, -np.inf], np.nan)
    missing_counts = X.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    # print(missing_counts)  # 缺失值都集中在single_perf_rel_i字段

    return X, y


# 测试在超参数组合下模型的表现
def train_cv_lgb(X, y, n_splits=5, params=None, seed=42, fast_mode=True):
    if params is None:
        # 构建默认超参数
        params = {
            'objective': 'regression',
            'verbosity': -1,  # 静默输出
            'random_state': seed,
            'n_estimators': 300 if not fast_mode else 100,
            'learning_rate': 0.05,
            'num_leaves': 31,  # 树中叶子节点个数
            'n_jobs': -1  # 并行线程数
        }

    # KFold 设置与初始化 OOF 容器
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)  # 随机打乱划分数据，n_splits = 5 折交叉验证
    oof_preds = np.zeros(len(y), dtype=float)  # 计算 OOF 指标 len(oof_preds) = len(y) = 900
    fold_metrics = []  # 按折保存每折 RMSE/MAE/R²
    models = []  # 保存每折训练出的 LGBMRegressor 对象

    # 折内训练循环   
    # # 打印每一折的数据划分情况
    # for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y), 1):
    #     print(f"Fold {fold}: Train indices {tr_idx}, Test indices {te_idx}")
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y), 1):
        # .iloc 根据位置选行
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)  # 在train上拟合模型
        preds = model.predict(X_te)  # 得到test集上数据的预测值
        oof_preds[te_idx] = preds
        # rmse = mean_squared_error(y_te, preds, squared=False)
        mse = mean_squared_error(y_te, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_te, preds)
        r2 = r2_score(y_te, preds)
        fold_metrics.append({'fold': fold, 'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)})
        models.append(model)
        print(f"Fold {fold}: RMSE={rmse:.6f}, MAE={mae:.6f}, R2={r2:.6f}")

    # OOF metrics
    # oof_rmse = mean_squared_error(y, oof_preds, squared=False)
    oof_mse = mean_squared_error(y, oof_preds)
    oof_rmse = np.sqrt(oof_mse)
    oof_mae = mean_absolute_error(y, oof_preds)
    oof_r2 = r2_score(y, oof_preds)
    print("OOF RMSE (log-target):", oof_rmse, 'MAE:', oof_mae, "R2:", oof_r2)

    # 额外在PPL空间报告指标
    try:
        true_ppl = np.exp(y)
        pred_ppl = np.exp(oof_preds)
        print("OOF RMSE(PPL):", mean_squared_error(true_ppl, pred_ppl, squared=False),  # 均方根误差
              "MAE:", mean_absolute_error(true_ppl, pred_ppl),
              "R2:", r2_score(true_ppl, pred_ppl))  # 判定系数，越接近于1越好
    except Exception:
        pass
    # modes: 训练的每折 LGBMRegressor 对象列表
    # oof_pres: 每个样本在它所在的 test fold 上的预测值 (log-target)
    # fold_metrics: 每折的指标列表 (字典列表形式)
    return models, oof_preds, fold_metrics


# # 对每一种model_name做一次“留一模型作为测试集”的评估
# # 在剩下的模型数据上训练模型，再在被留出的整个model上测试，用于评估回归器跨不同模型（不同层数/hidden_size）泛化的能力
# def leave_one_model_out_test(X, y, model_name_col='model_name', params=None, seed=42):

#     if model_name_col not in X.columns:
#         print("No model_name column found; skipping leave-one-model-out.")
#         return None

#     unique_models = np.unique(X[model_name_col])  # 收集所有不同的 model_name 值
#     results = []

#     for m in unique_models:  # 对每个model做一次单独的训练-测试循环，留出该model的所有行作为测试集
#         train_mask = X[model_name_col] != m
#         test_mask = X[model_name_col] == m
#         if test_mask.sum() == 0:
#             continue
#         # 构造训练/测试矩阵并drop model_name 列
#         X_tr = X.loc[train_mask].drop(columns=[model_name_col], errors='ignore')  # X train
#         y_tr = y.loc[train_mask]  # y train
#         X_te = X.loc[test_mask].drop(columns=[model_name_col], errors='ignore')  # X test
#         y_te = y.loc[test_mask]  # y test
#         # 模型训练与预测
#         model = lgb.LGBMRegressor(**(params or {}))
#         model.fit(X_tr, y_tr)
#         preds = model.predict(X_te)
#         # 计算评估指标
#         # rmse = mean_squared_error(y_te, preds, squared=False)
#         mse = mean_squared_error(y_te, preds)
#         rmse = np.sqrt(mse)
#         mae = mean_absolute_error(y_te, preds)
#         r2 = r2_score(y_te, preds)

#         results.append({
#             'left_out_model': m,
#             'rmse': float(rmse),
#             'mae': float(mae),
#             'r2': float(r2),
#             'n_test': int(test_mask.sum())
#         })

#         print("Left-out", m, '-> RMSE:', rmse, "MAE:", mae, "R2:", r2, "n_test:", int(test_mask.sum()))

#     return results


def train_final_and_save(X, y, params=None, out_model_path='./results/lgb_model.joblib'):
    model = lgb.LGBMRegressor(**(params or {}))
    model.fit(X, y)
    joblib.dump(model, out_model_path)
    print("Saved final model to", out_model_path)
    return model


# def bootstrap_uncertainty(X, y, rows_to_predict, n_boot=30, params=None, seed=42):
#     preds_all = []
#     rng = np.random.RandomState(seed)
#     n = len(x)
#     for b in range(n_boot):
#         idxs = rng.choice(n, size=n, replace=True)
#         Xb, yb = X.iloc[idxs], y.iloc[idxs]
#         model = lgb.LGBMRegressor(**(params or {}))
#         model.fit(Xb, yb)
#         preds_b = model.predict(rows_to_predict)
#         preds_all.append(preds_b)
#     preds_all = np.vstack(preds_all)
#     mean_pred = preds_all.mean(axis=0)
#     lower = np.percentile(preds_all, 2.5, axis=0)
#     upper = np.percentile(preds_all, 97.5, axxis=0)
#     return mean_pred, lower, upper


if __name__ == "__main__":

    df = load_data("./dataset/feature_data.csv")
    X, y = prepare_features(df)
    print("Feature count:", X.shape[1])  # 252
    
    # CV training
    params = {
        'objective': 'regression', 
        'verbosity': -1, 
        'random_state': 42,
        'n_estimators': 100, 
        'learning_rate': 0.05, 
        'num_leaves': 31, 
        'n_jobs': -1
    }
    models, oof_preds, fold_metrics = train_cv_lgb(X, y, n_splits=5, params=params, seed=42, fast_mode=True)
    pd.DataFrame(fold_metrics).to_csv('./results/cv_metrics.csv', index=False)
    pd.DataFrame({'y_true_log': y, 'y_pred_log': oof_preds, 'y_true_ppl': np.exp(y), 'y_pred_ppl': np.exp(oof_preds)}).to_csv('./results/cv_predictions.csv', index=False)
    
    

    # # optional leave-one-model-out
    # loo_res = None
    # if 'model_name' in X.columns:
    #     loo_res = leave_one_model_out_test(X.copy(), y.copy(), model_name_col='model_name', params=params, seed=42)
    #     if loo_res:
    #         pd.DataFrame(loo_res).to_csv('./results/leave_one_model_out.csv', index=False)

    # final model
    final_model = train_final_and_save(X, y, params=params, out_model_path="./results/lgb_model.joblib")
    
    # # feature importance
    # fi = pd.DataFrame({'feature': X.columns, 'importance': final_model.feature_importances_}).sort_values('importance', ascending=False)
    # fi.to_csv('./results/feature_importances.csv', index=False)
    # print("Wrote feature_importances.csv, cv_predictions.csv, cv_metrics.csv")
