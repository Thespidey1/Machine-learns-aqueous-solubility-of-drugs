from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

X_train_and_valid = pd.DataFrame(
    pd.read_excel('../X_train_and_valid.xlsx')).values
y_train_and_valid = pd.DataFrame(
    pd.read_excel('../y_train_and_valid.xlsx')).values
y_train_and_valid = y_train_and_valid.ravel()

scaler = StandardScaler()
scaler = scaler.fit(X_train_and_valid)
X_train_and_valid = scaler.transform(X_train_and_valid)


def xgb_parameters():
    """模型调参过程"""

    params = {'max_depth': range(5, 21, 1),
              'learning_rate': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.01, 0.02, 0.03]}

    fine_params = {'n_estimators': 500}
    return params, fine_params


def model_adjust_parameters(cv_params, other_params):
    """模型调参"""

    model = xgb.XGBRegressor(**other_params)

    optimized_param = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                                   verbose=3,
                                   n_jobs=-1)

    optimized_param.fit(X_train_and_valid, y_train_and_valid)

    means = optimized_param.cv_results_['mean_test_score']
    params = optimized_param.cv_results_['params']
    std = optimized_param.cv_results_['std_test_score']
    for mean, param, std in zip(means, params, std):
        print("mean_score: %f,  params: %r, std: %r" % (mean, param, std))

    print('参数的最佳取值：{0}'.format(optimized_param.best_params_))

    print('最佳模型得分:{0}'.format(optimized_param.best_score_))

    parameters_score = pd.DataFrame(data=[params, means, std])

    parameters_score.to_excel('./opt_2.xlsx', index=False)

    print('Optimization finished')


if __name__ == '__main__':
    """
        模型调参
        调参策略：网格搜索、随机搜索、启发式搜索
        补充：此处采用启发式搜索，逐个或逐类参数调整，避免所有参数一起调整导致模型训练复杂度过高
    """

    adj_params, fixed_params = xgb_parameters()

    model_adjust_parameters(adj_params, fixed_params)
