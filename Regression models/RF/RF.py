from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, recall_score, \
    precision_score, f1_score
import scikitplot as skplt
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import pickle

X_train_and_valid = pd.DataFrame(
    pd.read_excel('../X_train_and_valid.xlsx')).values
y_train_and_valid = pd.DataFrame(
    pd.read_excel('../y_train_and_valid.xlsx')).values
y_train_and_valid = y_train_and_valid.ravel()

folds = KFold(n_splits=5, shuffle=True)
i = 1
writer_y_fit = pd.ExcelWriter('y_fit.xlsx')
writer_y_Test = pd.ExcelWriter('y_Test.xlsx')
writer_y_fit_SC_1 = pd.ExcelWriter('y_fit_SC_1.xlsx')
writer_y_SC_1 = pd.ExcelWriter('y_SC_1.xlsx')
writer_y_fit_SC_2 = pd.ExcelWriter('y_fit_SC_2.xlsx')
writer_y_SC_2 = pd.ExcelWriter('y_SC_2.xlsx')
writer_y_fit_Drugbank = pd.ExcelWriter('y_fit_Drugbank.xlsx')
writer_y_Drugbank = pd.ExcelWriter('y_Drugbank.xlsx')
for trn_idx, test_idx in folds.split(X_train_and_valid, y_train_and_valid):
    X_Train, y_Train = X_train_and_valid[trn_idx, :], y_train_and_valid[trn_idx]
    X_Test, y_Test = X_train_and_valid[test_idx, :], y_train_and_valid[test_idx]

    scaler = StandardScaler()
    scaler = scaler.fit(X_Train)
    X_Train = scaler.transform(X_Train)
    X_Test = scaler.transform(X_Test)

    model = RandomForestRegressor(n_estimators=400, max_depth=50, min_samples_leaf=1, min_samples_split=2, n_jobs=-1)
    model.fit(X_Train, y_Train)

    y_pred = model.predict(X_Test)

    y_fit = pd.DataFrame(y_pred)

    y_test = pd.DataFrame(y_Test)

    MAE = mean_absolute_error(y_Test, y_pred)
    print('MAE：%.4f' % MAE)
    RMSE = math.sqrt(mean_squared_error(y_Test, y_pred))
    print('RMSE：%.4f' % RMSE)
    r2 = r2_score(y_Test, y_pred)
    print('r2：%.4f\n' % r2)
    y_fit.to_excel(writer_y_fit, index=False, sheet_name='%.0f' % i)
    y_test.to_excel(writer_y_Test, index=False, sheet_name='%.0f' % i)
    i = i + 1
writer_y_fit.close()
writer_y_Test.close()

for j in range(1, 6, 1):
    X_train_and_valid = pd.DataFrame(
        pd.read_excel('../X_train_and_valid.xlsx')).values
    y_train_and_valid = pd.DataFrame(
        pd.read_excel('../y_train_and_valid.xlsx')).values
    y_train_and_valid = y_train_and_valid.ravel()

    X_SC_1 = pd.DataFrame(
        pd.read_excel('../X_SC_R_1.xlsx')).values
    y_SC_1 = pd.DataFrame(
        pd.read_excel('../y_SC_R_1.xlsx')).values
    y_SC_1 = y_SC_1.ravel()

    X_SC_2 = pd.DataFrame(
        pd.read_excel('../X_SC_R_2.xlsx')).values
    y_SC_2 = pd.DataFrame(
        pd.read_excel('../y_SC_R_2.xlsx')).values
    y_SC_2 = y_SC_2.ravel()

    X_Drugbank = pd.DataFrame(
        pd.read_excel('../X_Drugbank_R.xlsx')).values
    y_Drugbank = pd.DataFrame(
        pd.read_excel('../y_Drugbank_R.xlsx')).values
    y_Drugbank = y_Drugbank.ravel()

    scaler = StandardScaler()
    scaler = scaler.fit(X_train_and_valid)
    X_train_and_valid = scaler.transform(X_train_and_valid)
    X_SC_1 = scaler.transform(X_SC_1)
    X_SC_2 = scaler.transform(X_SC_2)
    X_Drugbank = scaler.transform(X_Drugbank)

    model = RandomForestRegressor(n_estimators=400, max_depth=50, min_samples_leaf=1, min_samples_split=2, n_jobs=-1)
    model.fit(X_train_and_valid, y_train_and_valid)

    y_pred_SC_1 = model.predict(X_SC_1)
    y_fit_SC_1 = pd.DataFrame(y_pred_SC_1)
    y_fit_SC_1.to_excel('./y_fit_SC_1.xlsx', index=False)
    y_SC_1 = pd.DataFrame(y_SC_1)
    y_SC_1.to_excel('./y_SC_1.xlsx', index=False)
    MAE_SC_1 = mean_absolute_error(y_SC_1, y_pred_SC_1)
    print('MAE_SC_1：%.4f' % MAE_SC_1)
    RMSE_SC_1 = math.sqrt(mean_squared_error(y_SC_1, y_pred_SC_1))
    print('RMSE_SC_1：%.4f' % RMSE_SC_1)
    r2_SC_1 = r2_score(y_SC_1, y_pred_SC_1)
    print('r2_SC_1：%.4f\n' % r2_SC_1)

    y_pred_SC_2 = model.predict(X_SC_2)
    y_fit_SC_2 = pd.DataFrame(y_pred_SC_2)
    y_fit_SC_2.to_excel('./y_fit_SC_2.xlsx', index=False)
    y_SC_2 = pd.DataFrame(y_SC_2)
    y_SC_2.to_excel('./y_SC_2.xlsx', index=False)
    MAE_SC_2 = mean_absolute_error(y_SC_2, y_pred_SC_2)
    print('MAE_SC_2：%.4f' % MAE_SC_2)
    RMSE_SC_2 = math.sqrt(mean_squared_error(y_SC_2, y_pred_SC_2))
    print('RMSE_SC_2：%.4f' % RMSE_SC_2)
    r2_SC_2 = r2_score(y_SC_2, y_pred_SC_2)
    print('r2_SC_2：%.4f\n' % r2_SC_2)

    y_pred_Drugbank = model.predict(X_Drugbank)
    y_fit_Drugbank = pd.DataFrame(y_pred_Drugbank)
    y_fit_Drugbank.to_excel('./y_fit_Drugbank_R.xlsx', index=False)
    y_Drugbank = pd.DataFrame(y_Drugbank)
    y_Drugbank.to_excel('./y_Drugbank_R.xlsx', index=False)
    MAE_Drugbank = mean_absolute_error(y_Drugbank, y_pred_Drugbank)
    print('MAE_Drugbank：%.4f' % MAE_Drugbank)
    RMSE_Drugbank = math.sqrt(mean_squared_error(y_Drugbank, y_pred_Drugbank))
    print('RMSE__Drugbank：%.4f' % RMSE_Drugbank)
    r2_Drugbank = r2_score(y_Drugbank, y_pred_Drugbank)
    print('r2_Drugbank：%.4f\n' % r2_Drugbank)

    y_fit_SC_1.to_excel(writer_y_fit_SC_1, index=False, sheet_name='%.0f' % j)
    y_SC_1.to_excel(writer_y_SC_1, index=False, sheet_name='%.0f' % j)
    y_fit_SC_2.to_excel(writer_y_fit_SC_2, index=False, sheet_name='%.0f' % j)
    y_SC_2.to_excel(writer_y_SC_2, index=False, sheet_name='%.0f' % j)
    y_fit_Drugbank.to_excel(writer_y_fit_Drugbank, index=False, sheet_name='%.0f' % j)
    y_Drugbank.to_excel(writer_y_Drugbank, index=False, sheet_name='%.0f' % j)

writer_y_fit_SC_1.close()
writer_y_SC_1.close()
writer_y_fit_SC_2.close()
writer_y_SC_2.close()
writer_y_fit_Drugbank.close()
writer_y_Drugbank.close()
