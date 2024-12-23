import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

X_train_and_valid = pd.DataFrame(
    pd.read_excel('../X_train_and_valid.xlsx')).values
y_train_and_valid = pd.DataFrame(
    pd.read_excel('../y_train_and_valid.xlsx')).values
y_train_and_valid = y_train_and_valid.ravel()
i = 1
writer_random_num = pd.ExcelWriter('random_num.xlsx')
folds = KFold(n_splits=5, shuffle=True)
for trn_idx, test_idx in folds.split(X_train_and_valid, y_train_and_valid):
    test_idx = test_idx + 1
    test_idx = pd.DataFrame(test_idx)
    test_idx.to_excel(writer_random_num, index=False, sheet_name='%.0f' % i)
    i = i + 1
writer_random_num.close()
