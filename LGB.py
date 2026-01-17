import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import optuna


BASE_OUTPUT_PATH = 'output/All-LGB-Re'
os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

data = pd.read_csv('Data/All - Re.csv')
X = data.iloc[:, 1:-1]  # 特征 (排除第一列序号和最后一列标签)
y = data.iloc[:, -1]  # 标签

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_SEED)

feature_names = X.columns.tolist()

X_train.to_csv(f'{BASE_OUTPUT_PATH}/X_train.csv', index=False)
X_test.to_csv(f'{BASE_OUTPUT_PATH}/X_test.csv', index=False)
y_train.to_csv(f'{BASE_OUTPUT_PATH}/y_train.csv', index=False)
y_test.to_csv(f'{BASE_OUTPUT_PATH}/y_test.csv', index=False)

# 标准化处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 训练集标准化
X_test_scaled = scaler.transform(X_test)  # 测试集标准化

LGB = LGBMRegressor(random_state=RANDOM_SEED)
kf = KFold(10, shuffle=True, random_state=RANDOM_SEED)


# 超参数空间定义,初次为较大范围的参数空间，后续根据情况调整
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    num_leaves = trial.suggest_int('num_leaves', 2, 256)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.2)
    min_split_gain = trial.suggest_float('min_split_gain', 0.0, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-10, 10.0)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-10, 10.0)
    feature_fraction = trial.suggest_float('feature_fraction', 0.4, 1.0)
    bagging_freq = trial.suggest_int('bagging_freq', 0, 7)
    min_child_samples = trial.suggest_int('min_child_samples', 1, 100)

    LGB_model = LGBMRegressor(n_estimators=n_estimators,
                              num_leaves=num_leaves,
                              learning_rate=learning_rate,
                              min_split_gain=min_split_gain,
                              reg_alpha=reg_alpha,
                              reg_lambda=reg_lambda,
                              feature_fraction=feature_fraction,
                              bagging_freq=bagging_freq,
                              min_child_samples=min_child_samples,
                              random_state=42)

    score = cross_val_score(LGB_model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
    return -score


# 创建Optuna的Study对象并进行优化
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=50)

# 获取最佳参数
best_params = study.best_params
print(f'Best parameters: {best_params}')

with open(f'{BASE_OUTPUT_PATH}/best_params.txt', 'w') as f:
    f.write(f"Best Parameters: {best_params}\n")

# 使用最佳参数训练模型
best_LGB = LGBMRegressor(
    n_estimators=best_params['n_estimators'],
    num_leaves=best_params['num_leaves'],
    learning_rate=best_params['learning_rate'],
    min_split_gain=best_params['min_split_gain'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    feature_fraction=best_params['feature_fraction'],
    bagging_freq=best_params['bagging_freq'],
    min_child_samples=best_params['min_child_samples'],
)


# 训练模型
best_LGB.fit(X_train_scaled, y_train)

y_train_pred = best_LGB.predict(X_train_scaled)
y_test_pred = best_LGB.predict(X_test_scaled)

# 计算评价指标
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

results = [
    f"train R2: {r2_train}, test R2: {r2_test}",
    f"train RMSE: {rmse_train}, test RMSE: {rmse_test}",
    f"train MAE: {mae_train}, test MAE: {mae_test}"
]

# 打印结果到控制台
for result in results:
    print(result)

# 将结果保存到文件
with open(f'{BASE_OUTPUT_PATH}/results.txt', 'w') as f:
    for result in results:
        f.write(result + '\n')
