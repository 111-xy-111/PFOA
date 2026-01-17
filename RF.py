import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import optuna
import os

BASE_OUTPUT_PATH = 'output/All-RF-Re'
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

rf = RandomForestRegressor(random_state=RANDOM_SEED)
kf = KFold(10, shuffle=True, random_state=RANDOM_SEED)

# 超参数空间定义,初次为较大范围的参数空间，后续根据情况调整
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 50)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.7, 0.8, 0.9])

    rf_model = RandomForestRegressor(n_estimators=n_estimators,
                                     max_depth=max_depth,
                                     min_samples_split=min_samples_split,
                                     min_samples_leaf=min_samples_leaf,
                                     max_features=max_features,
                                     random_state=42)

    score = cross_val_score(rf_model, X_train_scaled, y_train, cv=kf, scoring='neg_mean_squared_error').mean()
    return -score


# 创建Optuna的Study对象并进行优化
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=50)

best_params = study.best_params
print(f'Best parameters: {best_params}')
print(f'Best parameters: {best_params}')

with open(f'{BASE_OUTPUT_PATH}/best_params.txt', 'w') as f:
    f.write(f"Best Parameters: {best_params}\n")

# 使用最佳参数训练模型
best_rf = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf']
)

# 训练模型
best_rf.fit(X_train_scaled, y_train)

# 评估最佳模型（仅在测试集上进行）
y_train_pred = best_rf.predict(X_train_scaled)
y_test_pred = best_rf.predict(X_test_scaled)

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


for result in results:
    print(result)

# 将结果保存到文件
with open(f'{BASE_OUTPUT_PATH}/results.txt', 'w') as f:
    for result in results:
        f.write(result + '\n')
