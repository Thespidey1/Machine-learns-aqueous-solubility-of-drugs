from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

X_train_and_valid = pd.DataFrame(
    pd.read_excel('../X_train_and_valid.xlsx')).values
y_train_and_valid = pd.DataFrame(
    pd.read_excel('../y_train_and_valid.xlsx')).values
y_train_and_valid = y_train_and_valid.ravel()

scaler = StandardScaler()
scaler = scaler.fit(X_train_and_valid)
X_train_and_valid = scaler.transform(X_train_and_valid)


def rf_parameters():
    params = {'max_depth': range(30, 110, 10),
              'n_estimators': range(200, 600, 100)}

    fine_params = {
        'min_samples_leaf': 1, 'min_samples_split': 2,
    }
    return params, fine_params


def model_adjust_parameters(cv_params, other_params):
    model = RandomForestRegressor(**other_params)

    optimized_param = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                                   verbose=3, n_jobs=-1)

    optimized_param.fit(X_train_and_valid, y_train_and_valid)

    means = optimized_param.cv_results_['mean_test_score']
    params = optimized_param.cv_results_['params']
    std = optimized_param.cv_results_['std_test_score']
    for mean, param, std in zip(means, params, std):
        print("mean_score: %f,  params: %r, std: %r" % (mean, param, std))

    print('参数的最佳取值：{0}'.format(optimized_param.best_params_))

    print('最佳模型得分:{0}'.format(optimized_param.best_score_))

    parameters_score = pd.DataFrame(data=[params, means, std])

    parameters_score.to_excel('./opt_3.xlsx', index=False)

    print('Optimization finished')


if __name__ == '__main__':
    adj_params, fixed_params = rf_parameters()

    model_adjust_parameters(adj_params, fixed_params)
