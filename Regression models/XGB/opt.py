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

    params = {'max_depth': range(5, 21, 1),
              'learning_rate': [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.01, 0.02, 0.03]}

    fine_params = {'n_estimators': 500}
    return params, fine_params


def model_adjust_parameters(cv_params, other_params):

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

    print('best parameters：{0}'.format(optimized_param.best_params_))

    print('best score:{0}'.format(optimized_param.best_score_))

    print('Optimization finished')


if __name__ == '__main__':

    adj_params, fixed_params = xgb_parameters()

    model_adjust_parameters(adj_params, fixed_params)
