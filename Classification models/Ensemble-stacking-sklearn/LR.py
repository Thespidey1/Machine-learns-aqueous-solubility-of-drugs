import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, average_precision_score
import scikitplot as skplt
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import lightgbm
from sklearn.model_selection import KFold
import matplotlib as mpl

import warnings

warnings.filterwarnings("ignore")

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

X_train_and_valid = pd.DataFrame(
    pd.read_excel('../X_train_and_valid_0.01.xlsx')).values
y_train_and_valid = pd.DataFrame(
    pd.read_excel('../y_train_and_valid.xlsx')).values
y_train_and_valid = y_train_and_valid.ravel()

folds = KFold(n_splits=5, shuffle=True)
i = 1
writer_y_fit = pd.ExcelWriter('y_fit.xlsx')
writer_y_Test = pd.ExcelWriter('y_Test.xlsx')
writer_y_score = pd.ExcelWriter('y_score.xlsx')

writer_y_fit_SC_1 = pd.ExcelWriter('y_fit_SC_1.xlsx')
writer_y_SC_1 = pd.ExcelWriter('y_SC_1.xlsx')
writer_y_SC_1_score = pd.ExcelWriter('y_SC_1_score.xlsx')

writer_y_fit_SC_2 = pd.ExcelWriter('y_fit_SC_2.xlsx')
writer_y_SC_2 = pd.ExcelWriter('y_SC_2.xlsx')
writer_y_SC_2_score = pd.ExcelWriter('y_SC_2_score.xlsx')

writer_y_fit_Drugbank = pd.ExcelWriter('y_fit_Drugbank.xlsx')
writer_y_Drugbank = pd.ExcelWriter('y_Drugbank.xlsx')
writer_y_Drugbank_score = pd.ExcelWriter('y_Drugbank_score.xlsx')

for trn_idx, test_idx in folds.split(X_train_and_valid, y_train_and_valid):
    X_Train, y_Train = X_train_and_valid[trn_idx, :], y_train_and_valid[trn_idx]
    X_Test, y_Test = X_train_and_valid[test_idx, :], y_train_and_valid[test_idx]

    scaler = StandardScaler()
    scaler = scaler.fit(X_Train)
    X_Train = scaler.transform(X_Train)
    X_Test = scaler.transform(X_Test)

    lgb = lightgbm.LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.1, force_col_wise=True,
                                  n_jobs=-1, verbose=-1, num_leaves=30)
    xgboost = xgb.XGBClassifier(n_estimators=300, max_depth=13, learning_rate=0.08)
    rf = RandomForestClassifier(n_estimators=400, max_depth=45, min_samples_leaf=1, min_samples_split=2)
    svc = svm.SVC(kernel='rbf', probability=True, C=1, gamma=0.07)

    model = StackingClassifier(estimators=[('lightgbm', lgb), ('xgb', xgboost), ('rf', rf), ('svc', svc)],
                               final_estimator=LogisticRegression(n_jobs=-1, fit_intercept=False, max_iter=1000, C=0.3),
                               n_jobs=-1, cv=5)
    model.fit(X_Train, y_Train)

    y_pred = model.predict(X_Test)
    y_proba = model.predict_proba(X_Test)
    cm = confusion_matrix(y_Test, y_pred)

    y_fit = pd.DataFrame(y_pred)
    score = pd.DataFrame(y_proba)
    y_test = pd.DataFrame(y_Test)

    accuracy = accuracy_score(y_Test, y_pred)
    print('Accuracy：%.4f' % accuracy)
    n_classes = len(set(y_train_and_valid))

    if n_classes == 2:
        precision = precision_score(y_Test, y_pred)
    else:
        precision = precision_score(y_Test, y_pred, average='macro')
    print('Precision：%.4f' % precision)

    if n_classes == 2:
        recall = recall_score(y_Test, y_pred)
    else:
        recall = recall_score(y_Test, y_pred, average='macro')
    print('Recall：%.4f' % recall)

    if n_classes == 2:
        f1 = f1_score(y_Test, y_pred)
    else:
        f1 = f1_score(y_Test, y_pred, average='macro')
    print('F1：%.4f' % f1)

    if n_classes == 2:
        auc = roc_auc_score(y_Test, y_pred)
    else:
        auc = roc_auc_score(y_Test, y_proba, multi_class='ovr', average="macro", )

    print('AUC：%.4f' % auc)

    if n_classes == 2:
        AP = average_precision_score(y_Test, y_pred)
    else:
        AP = average_precision_score(y_Test, y_proba, average="macro")
    print('AP：%.4f\n' % AP)

    y_fit.to_excel(writer_y_fit, index=False, sheet_name='%.0f' % i)
    y_test.to_excel(writer_y_Test, index=False, sheet_name='%.0f' % i)
    score.to_excel(writer_y_score, index=False, sheet_name='%.0f' % i)
    i = i + 1
writer_y_fit.close()
writer_y_Test.close()
writer_y_score.close()

for j in range(1, 6, 1):
    X_train_and_valid = pd.DataFrame(
        pd.read_excel('../X_train_and_valid_0.01.xlsx')).values
    y_train_and_valid = pd.DataFrame(
        pd.read_excel('../y_train_and_valid.xlsx')).values
    y_train_and_valid = y_train_and_valid.ravel()

    X_SC_1 = pd.DataFrame(
        pd.read_excel('../X_SC_C_1_0.01.xlsx')).values
    y_SC_1 = pd.DataFrame(
        pd.read_excel('../y_SC_C_1.xlsx')).values
    y_SC_1 = y_SC_1.ravel()

    X_SC_2 = pd.DataFrame(
        pd.read_excel('../X_SC_C_2_0.01.xlsx')).values
    y_SC_2 = pd.DataFrame(
        pd.read_excel('../y_SC_C_2.xlsx')).values
    y_SC_2 = y_SC_2.ravel()

    X_Drugbank = pd.DataFrame(
        pd.read_excel('../X_Drugbank_C_0.01.xlsx')).values
    y_Drugbank = pd.DataFrame(
        pd.read_excel('../y_Drugbank_C.xlsx')).values
    y_Drugbank = y_Drugbank.ravel()

    scaler = StandardScaler()
    scaler = scaler.fit(X_train_and_valid)
    X_train_and_valid = scaler.transform(X_train_and_valid)
    X_SC_1 = scaler.transform(X_SC_1)
    X_SC_2 = scaler.transform(X_SC_2)
    X_Drugbank = scaler.transform(X_Drugbank)

    lgb = lightgbm.LGBMClassifier(n_estimators=500, max_depth=5, learning_rate=0.1, force_col_wise=True,
                                  n_jobs=-1, verbose=-1, num_leaves=30)
    xgboost = xgb.XGBClassifier(n_estimators=300, max_depth=13, learning_rate=0.08)
    rf = RandomForestClassifier(n_estimators=400, max_depth=45, min_samples_leaf=1, min_samples_split=2)
    svc = svm.SVC(kernel='rbf', probability=True, C=1, gamma=0.07)

    model = StackingClassifier(estimators=[('lightgbm', lgb), ('xgb', xgboost), ('rf', rf), ('svc', svc)],
                               final_estimator=LogisticRegression(n_jobs=-1, fit_intercept=False, max_iter=1000, C=0.3),
                               n_jobs=-1, cv=5)
    model.fit(X_train_and_valid, y_train_and_valid)

    y_pred_SC_1 = model.predict(X_SC_1)
    y_proba_SC_1 = model.predict_proba(X_SC_1)
    cm_SC_1 = confusion_matrix(y_SC_1, y_pred_SC_1)
    y_fit_SC_1 = pd.DataFrame(y_pred_SC_1)
    score_SC_1 = pd.DataFrame(y_proba_SC_1)
    y_SC_1 = pd.DataFrame(y_SC_1)
    accuracy_SC_1 = accuracy_score(y_SC_1, y_pred_SC_1)
    print('Accuracy_SC_1：%.4f' % accuracy_SC_1)
    n_classes = len(set(y_train_and_valid))

    if n_classes == 2:
        precision_SC_1 = precision_score(y_SC_1, y_pred_SC_1)
    else:
        precision_SC_1 = precision_score(y_SC_1, y_pred_SC_1, average='macro')
    print('Precision_SC_1：%.4f' % precision_SC_1)

    if n_classes == 2:
        recall_SC_1 = recall_score(y_SC_1, y_pred_SC_1)
    else:
        recall_SC_1 = recall_score(y_SC_1, y_pred_SC_1, average='macro')
    print('Recall_SC_1：%.4f' % recall_SC_1)

    if n_classes == 2:
        f1_SC_1 = f1_score(y_SC_1, y_pred_SC_1)
    else:
        f1_SC_1 = f1_score(y_SC_1, y_pred_SC_1, average='macro')
    print('F1_SC_1：%.4f\n' % f1_SC_1)

    y_pred_SC_2 = model.predict(X_SC_2)
    y_proba_SC_2 = model.predict_proba(X_SC_2)
    cm_SC_2 = confusion_matrix(y_SC_2, y_pred_SC_2)

    y_fit_SC_2 = pd.DataFrame(y_pred_SC_2)
    score_SC_2 = pd.DataFrame(y_proba_SC_2)
    y_SC_2 = pd.DataFrame(y_SC_2)

    accuracy_SC_2 = accuracy_score(y_SC_2, y_pred_SC_2)
    print('Accuracy_SC_2：%.4f' % accuracy_SC_2)
    n_classes = len(set(y_train_and_valid))

    if n_classes == 2:
        precision_SC_2 = precision_score(y_SC_2, y_pred_SC_2)
    else:
        precision_SC_2 = precision_score(y_SC_2, y_pred_SC_2, average='macro')
    print('Precision_SC_2：%.4f' % precision_SC_2)

    if n_classes == 2:
        recall_SC_2 = recall_score(y_SC_2, y_pred_SC_2)
    else:
        recall_SC_2 = recall_score(y_SC_2, y_pred_SC_2, average='macro')
    print('Recall_SC_2：%.4f' % recall_SC_2)

    if n_classes == 2:
        f1_SC_2 = f1_score(y_SC_2, y_pred_SC_2)
    else:
        f1_SC_2 = f1_score(y_SC_2, y_pred_SC_2, average='macro')
    print('F1_SC_2：%.4f\n' % f1_SC_2)

    y_pred_Drugbank = model.predict(X_Drugbank)
    y_proba_Drugbank = model.predict_proba(X_Drugbank)
    cm_Drugbank = confusion_matrix(y_Drugbank, y_pred_Drugbank)

    y_fit_Drugbank = pd.DataFrame(y_pred_Drugbank)
    score_Drugbank = pd.DataFrame(y_proba_Drugbank)
    y_Drugbank = pd.DataFrame(y_Drugbank)

    accuracy_Drugbank = accuracy_score(y_Drugbank, y_pred_Drugbank)
    print('Accuracy_Drugbank：%.4f' % accuracy_Drugbank)
    n_classes = len(set(y_train_and_valid))

    if n_classes == 2:
        precision_Drugbank = precision_score(y_Drugbank, y_pred_Drugbank)
    else:
        precision_Drugbank = precision_score(y_Drugbank, y_pred_Drugbank, average='macro')
    print('Precision_Drugbank：%.4f' % precision_Drugbank)

    if n_classes == 2:
        recall_Drugbank = recall_score(y_Drugbank, y_pred_Drugbank)
    else:
        recall_Drugbank = recall_score(y_Drugbank, y_pred_Drugbank, average='macro')
    print('Recall_Drugbank：%.4f' % recall_Drugbank)

    if n_classes == 2:
        f1_Drugbank = f1_score(y_Drugbank, y_pred_Drugbank)
    else:
        f1_Drugbank = f1_score(y_Drugbank, y_pred_Drugbank, average='macro')
    print('F1_Drugbank：%.4f' % f1_Drugbank)

    if n_classes == 2:
        auc_Drugbank = roc_auc_score(y_Drugbank, y_pred_Drugbank)
    else:
        auc_Drugbank = roc_auc_score(y_Drugbank, y_proba_Drugbank, multi_class='ovr', average="macro", )

    print('AUC_Drugbank：%.4f' % auc_Drugbank)

    if n_classes == 2:
        AP_Drugbank = average_precision_score(y_Drugbank, y_pred_Drugbank)
    else:
        AP_Drugbank = average_precision_score(y_Drugbank, y_proba_Drugbank, average="macro")

    print('AP_Drugbank：%.4f\n' % AP_Drugbank)

    y_fit_SC_1.to_excel(writer_y_fit_SC_1, index=False, sheet_name='%.0f' % j)
    y_SC_1.to_excel(writer_y_SC_1, index=False, sheet_name='%.0f' % j)
    score_SC_1.to_excel(writer_y_SC_1_score, index=False, sheet_name='%.0f' % j)
    y_fit_SC_2.to_excel(writer_y_fit_SC_2, index=False, sheet_name='%.0f' % j)
    y_SC_2.to_excel(writer_y_SC_2, index=False, sheet_name='%.0f' % j)
    score_SC_2.to_excel(writer_y_SC_2_score, index=False, sheet_name='%.0f' % j)
    y_fit_Drugbank.to_excel(writer_y_fit_Drugbank, index=False, sheet_name='%.0f' % j)
    y_Drugbank.to_excel(writer_y_Drugbank, index=False, sheet_name='%.0f' % j)
    score_Drugbank.to_excel(writer_y_Drugbank_score, index=False, sheet_name='%.0f' % j)

writer_y_fit_SC_1.close()
writer_y_SC_1.close()
writer_y_fit_SC_2.close()
writer_y_SC_2.close()
writer_y_fit_Drugbank.close()
writer_y_Drugbank.close()
writer_y_SC_1_score.close()
writer_y_SC_2_score.close()
writer_y_Drugbank_score.close()
