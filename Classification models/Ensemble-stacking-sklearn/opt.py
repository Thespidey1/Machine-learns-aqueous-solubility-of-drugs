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
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import math
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import lightgbm
from sklearn.model_selection import KFold
import matplotlib as mpl

import warnings

warnings.filterwarnings("ignore")

X_train_and_valid = pd.DataFrame(
    pd.read_excel('../X_train_and_valid_0.01.xlsx')).values
y_train_and_valid = pd.DataFrame(
    pd.read_excel('../y_train_and_valid.xlsx')).values
y_train_and_valid = y_train_and_valid.ravel()

scaler = StandardScaler()
scaler = scaler.fit(X_train_and_valid)
X_train_and_valid = scaler.transform(X_train_and_valid)


def lr_parameters():
    params = {'final_estimator': [LogisticRegression(n_jobs=-1, fit_intercept=False, C=0, max_iter=1000),
                                  LogisticRegression(n_jobs=-1, fit_intercept=False, C=0.0001, max_iter=1000),
                                  LogisticRegression(n_jobs=-1, fit_intercept=False, C=0.0003, max_iter=1000),
                                  LogisticRegression(n_jobs=-1, fit_intercept=False, C=0.001, max_iter=1000),
                                  LogisticRegression(n_jobs=-1, fit_intercept=False, C=0.003, max_iter=1000),
                                  LogisticRegression(n_jobs=-1, fit_intercept=False, C=0.01, max_iter=1000),
                                  LogisticRegression(n_jobs=-1, fit_intercept=False, C=0.03, max_iter=1000),
                                  LogisticRegression(n_jobs=-1, fit_intercept=False, C=0.1, max_iter=1000),
                                  LogisticRegression(n_jobs=-1, fit_intercept=False, C=0.3, max_iter=1000),
                                  LogisticRegression(n_jobs=-1, fit_intercept=False, C=1, max_iter=1000)]}
    fine_params = {}
    return params, fine_params


def model_adjust_parameters(cv_params, other_params):
    lgb = lightgbm.LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.1, force_col_wise=True,
                                  n_jobs=-1, verbose=-1, num_leaves=30)
    xgboost = xgb.XGBClassifier(n_estimators=300, max_depth=13, learning_rate=0.08)
    rf = RandomForestClassifier(n_estimators=400, max_depth=45, min_samples_leaf=1, min_samples_split=2)
    svc = svm.SVC(kernel='rbf', probability=True, C=1, gamma=0.07)
    model = StackingClassifier(estimators=[('lightgbm', lgb), ('xgb', xgboost), ('rf', rf), ('svc', svc)],
                               n_jobs=-1, cv=5)

    optimized_param = GridSearchCV(estimator=model, param_grid=cv_params, scoring='accuracy', cv=5,
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

    parameters_score.to_excel('./opt_1.xlsx', index=False)
    print('Optimization finished')


if __name__ == '__main__':
    adj_params, fixed_params = lr_parameters()
    model_adjust_parameters(adj_params, fixed_params)
