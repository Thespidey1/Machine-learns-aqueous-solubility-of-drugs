from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, average_precision_score
import scikitplot as skplt
import numpy as np

y_SC_1 = pd.read_excel('class_lasso_for_compute.xlsx', sheet_name=0)
y_pred_SC_1 = pd.read_excel('ex_class.xlsx', sheet_name=0)

accuracy_SC_1 = accuracy_score(y_SC_1, y_pred_SC_1)
print('Accuracy_SC_1：%.4f' % accuracy_SC_1)
n_classes = 7

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
print('F1_SC_1：%.4f' % f1_SC_1)

y_SC_2 = pd.read_excel('class_lasso_for_compute.xlsx', sheet_name=1)
y_pred_SC_2 = pd.read_excel('ex_class.xlsx', sheet_name=1)

accuracy_SC_2 = accuracy_score(y_SC_2, y_pred_SC_2)
print('Accuracy_SC_2：%.4f' % accuracy_SC_2)

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
print('F1_SC_2：%.4f' % f1_SC_2)

y_DrugBank = pd.read_excel('class_lasso_for_compute.xlsx', sheet_name=2)
y_pred_DrugBank = pd.read_excel('ex_class.xlsx', sheet_name=2)

accuracy_DrugBank = accuracy_score(y_DrugBank, y_pred_DrugBank)
print('Accuracy_DrugBank：%.4f' % accuracy_DrugBank)

if n_classes == 2:
    precision_DrugBank = precision_score(y_DrugBank, y_pred_DrugBank)
else:
    precision_DrugBank = precision_score(y_DrugBank, y_pred_DrugBank, average='macro')
print('Precision_DrugBank：%.4f' % precision_DrugBank)


if n_classes == 2:
    recall_DrugBank = recall_score(y_DrugBank, y_pred_DrugBank)
else:
    recall_DrugBank = recall_score(y_DrugBank, y_pred_DrugBank, average='macro')
print('Recall_DrugBank：%.4f' % recall_DrugBank)

if n_classes == 2:
    f1_DrugBank = f1_score(y_DrugBank, y_pred_DrugBank)
else:
    f1_DrugBank = f1_score(y_DrugBank, y_pred_DrugBank, average='macro')
print('F1_DrugBank：%.4f' % f1_DrugBank)

y_all = pd.read_excel('class_lasso_for_compute.xlsx', sheet_name=3)
y_pred_all = pd.read_excel('ex_class.xlsx', sheet_name=3)

accuracy_all = accuracy_score(y_all, y_pred_all)
print('Accuracy_all：%.4f' % accuracy_all)

if n_classes == 2:
    precision_all = precision_score(y_all, y_pred_all)
else:
    precision_all = precision_score(y_all, y_pred_all, average='macro')
print('Precision_all：%.4f' % precision_all)


if n_classes == 2:
    recall_all = recall_score(y_all, y_pred_all)
else:
    recall_all = recall_score(y_all, y_pred_all, average='macro')
print('Recall_all：%.4f' % recall_all)

if n_classes == 2:
    f1_all = f1_score(y_all, y_pred_all)
else:
    f1_all = f1_score(y_all, y_pred_all, average='macro')
print('F1_all：%.4f' % f1_all)
