from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scikitplot as skplt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import math
import numpy as np
import pickle
from sklearn.linear_model import Ridge
import lightgbm
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm

X_train_and_valid = pd.DataFrame(
    pd.read_excel('../X_train_and_valid.xlsx')).values
y_train_and_valid = pd.DataFrame(
    pd.read_excel('../y_train_and_valid.xlsx')).values
y_train_and_valid = y_train_and_valid.ravel()

scaler = StandardScaler()
scaler = scaler.fit(X_train_and_valid)
X_train_and_valid = scaler.transform(X_train_and_valid)


def ridge_parameters():
    params = {'final_estimator': [Ridge(fit_intercept=False, positive=True, alpha=1),
                                  Ridge(fit_intercept=False, positive=True, alpha=0.3),
                                  Ridge(fit_intercept=False, positive=True, alpha=0.1),
                                  Ridge(fit_intercept=False, positive=True, alpha=0.03),
                                  Ridge(fit_intercept=False, positive=True, alpha=0.01),
                                  Ridge(fit_intercept=False, positive=True, alpha=0.003),
                                  Ridge(fit_intercept=False, positive=True, alpha=0.001),
                                  Ridge(fit_intercept=False, positive=True, alpha=0.0003),
                                  Ridge(fit_intercept=False, positive=True, alpha=0.0001)]}
    fine_params = {}
    return params, fine_params


def model_adjust_parameters(cv_params, other_params):
    xgboost = xgb.XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.07)
    rf = RandomForestRegressor(n_estimators=400, max_depth=50, min_samples_leaf=1, min_samples_split=2)
    svr = svm.SVR(kernel='rbf', C=10, gamma=0.004)
    lgb = lightgbm.LGBMRegressor(n_estimators=3000, max_depth=5, learning_rate=0.07, force_col_wise=True, n_jobs=-1,
                                 verbose=-1, num_leaves=30)
    model = StackingRegressor(estimators=[('xgb', xgboost), ('rf', rf), ('svr', svr), ('lightgbm', lgb)], n_jobs=-1,
                              cv=5)
    optimized_param = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_squared_error', cv=5,
                                   verbose=3, n_jobs=-1)

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
    adj_params, fixed_params = ridge_parameters()
    model_adjust_parameters(adj_params, fixed_params)
