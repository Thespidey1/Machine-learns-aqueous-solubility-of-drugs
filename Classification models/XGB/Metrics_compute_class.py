import numpy
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, \
    recall_score, f1_score, average_precision_score
import scikitplot as skplt
import numpy as np

y_pred_SC_1 = []
y_proba_SC_1 = pd.DataFrame(
    pd.read_excel('./y_SC_1_score.xlsx', sheet_name=0)).values
y_SC_1 = []
for i in range(0, 5, 1):
    a = pd.DataFrame(
        pd.read_excel('./y_fit_SC_1.xlsx', sheet_name=i)).values
    a = a.flatten()
    b = pd.DataFrame(
        pd.read_excel('./y_SC_1_score.xlsx', sheet_name=i)).values
    c = pd.read_excel('./y_SC_1.xlsx', sheet_name=i).values
    c = c.flatten()
    y_pred_SC_1 = np.concatenate([y_pred_SC_1, a], axis=0)
    if i > 0:
        y_proba_SC_1 = np.concatenate([y_proba_SC_1, b], axis=0)
    y_SC_1 = np.concatenate([y_SC_1, c], axis=0)

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

actual_labels = set(np.unique(y_SC_1).tolist())

predicted_labels = set(range(y_proba_SC_1.shape[1]))

valid_labels = list(actual_labels.intersection(predicted_labels))
valid_labels = [int(label) for label in valid_labels]

y_proba_SC_1_filtered = y_proba_SC_1[:, valid_labels]
y_proba_SC_1_normalized = normalize(y_proba_SC_1_filtered, norm='l1', axis=1)

if len(valid_labels) == 2:
    auc_SC_1 = roc_auc_score(y_SC_1, y_pred_SC_1)
else:
    auc_SC_1 = roc_auc_score(y_SC_1, y_proba_SC_1_normalized, multi_class='ovr', average='macro', labels=valid_labels)
print('AUC_SC_1：%.4f' % auc_SC_1)

if len(valid_labels) == 2:
    AP_SC_1 = average_precision_score(y_SC_1, y_pred_SC_1)
else:
    AP_SC_1 = average_precision_score(y_SC_1, y_proba_SC_1_normalized, average='macro')
print('AP_SC_1：%.4f\n' % AP_SC_1)

y_SC_1 = y_SC_1.astype(int)

skplt.metrics.plot_roc(y_SC_1, y_proba_SC_1_normalized, figsize=(6.5, 5.8333), plot_micro=True, plot_macro=True,
                       classes_to_plot=None, title="")
plt.legend(prop={'family': 'Arial', 'weight': 'normal', 'size': 13})
plt.xlabel('False positive rate', fontsize=15, weight='normal', family='Arial')
plt.ylabel('True positive rate', fontsize=15, weight='normal', family='Arial')
plt.xticks(fontsize=15, weight='normal', family='Arial')
plt.yticks(fontsize=15, weight='normal', family='Arial')

plt.savefig('./ROC_curve_SC_1.png', dpi=1500)
plt.savefig('./ROC_curve_SC_1.pdf', dpi=1500)

y_pred_SC_2 = []
y_proba_SC_2 = pd.DataFrame(
    pd.read_excel('./y_SC_2_score.xlsx', sheet_name=0)).values
y_SC_2 = []
for i in range(0, 5, 1):
    a = pd.DataFrame(
        pd.read_excel('./y_fit_SC_2.xlsx', sheet_name=i)).values
    a = a.flatten()
    b = pd.DataFrame(
        pd.read_excel('./y_SC_2_score.xlsx', sheet_name=i)).values
    c = pd.read_excel('./y_SC_2.xlsx', sheet_name=i).values
    c = c.flatten()
    y_pred_SC_2 = np.concatenate([y_pred_SC_2, a], axis=0)
    if i > 0:
        y_proba_SC_2 = np.concatenate([y_proba_SC_2, b], axis=0)
    y_SC_2 = np.concatenate([y_SC_2, c], axis=0)

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

actual_labels = set(np.unique(y_SC_2).tolist())

predicted_labels = set(range(y_proba_SC_2.shape[1]))

valid_labels = list(actual_labels.intersection(predicted_labels))
valid_labels = [int(label) for label in valid_labels]

y_proba_SC_2_filtered = y_proba_SC_2[:, valid_labels]

y_proba_SC_2_normalized = normalize(y_proba_SC_2_filtered, norm='l1', axis=1)

if len(valid_labels) == 2:
    auc_SC_2 = roc_auc_score(y_SC_2, y_pred_SC_2)
else:
    auc_SC_2 = roc_auc_score(y_SC_2, y_proba_SC_2_normalized, multi_class='ovr', average='macro', labels=valid_labels)
print('AUC_SC_2：%.4f' % auc_SC_2)

if len(valid_labels) == 2:
    AP_SC_2 = average_precision_score(y_SC_2, y_pred_SC_2)
else:
    AP_SC_2 = average_precision_score(y_SC_2, y_proba_SC_2_normalized, average='macro')
print('AP_SC_2：%.4f\n' % AP_SC_2)

y_SC_2 = y_SC_2.astype(int)
skplt.metrics.plot_roc(y_SC_2, y_proba_SC_2_normalized, figsize=(6.5, 5.8333), plot_micro=True, plot_macro=True,
                       classes_to_plot=None, title="")
plt.legend(prop={'family': 'Arial', 'weight': 'normal', 'size': 13})
plt.xlabel('False positive rate', fontsize=15, weight='normal', family='Arial')
plt.ylabel('True positive rate', fontsize=15, weight='normal', family='Arial')
plt.xticks(fontsize=15, weight='normal', family='Arial')
plt.yticks(fontsize=15, weight='normal', family='Arial')

plt.savefig('./ROC_curve_SC_2.png', dpi=1500)
plt.savefig('./ROC_curve_SC_2.pdf', dpi=1500)

y_pred_DrugBank = []
y_proba_DrugBank = pd.DataFrame(
    pd.read_excel('./y_Drugbank_score.xlsx', sheet_name=0)).values
y_DrugBank = []
for i in range(0, 5, 1):
    a = pd.DataFrame(
        pd.read_excel('./y_fit_Drugbank.xlsx', sheet_name=i)).values
    a = a.flatten()
    b = pd.DataFrame(
        pd.read_excel('./y_Drugbank_score.xlsx', sheet_name=i)).values
    c = pd.read_excel('./y_Drugbank.xlsx', sheet_name=i).values
    c = c.flatten()
    y_pred_DrugBank = np.concatenate([y_pred_DrugBank, a], axis=0)
    if i > 0:
        y_proba_DrugBank = np.concatenate([y_proba_DrugBank, b], axis=0)
    y_DrugBank = np.concatenate([y_DrugBank, c], axis=0)

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

if n_classes == 2:
    auc_DrugBank = roc_auc_score(y_DrugBank, y_pred_DrugBank)
else:
    auc_DrugBank = roc_auc_score(y_DrugBank, y_proba_DrugBank, multi_class='ovr', average='macro')
print('AUC_DrugBank：%.4f' % auc_DrugBank)

if n_classes == 2:
    AP_DrugBank = average_precision_score(y_DrugBank, y_pred_DrugBank)
else:
    AP_DrugBank = average_precision_score(y_DrugBank, y_proba_DrugBank, average='macro')
print('AP_DrugBank：%.4f\n' % AP_DrugBank)

y_DrugBank = y_DrugBank.astype(int)
skplt.metrics.plot_roc(y_DrugBank, y_proba_DrugBank, figsize=(6.5, 5.8333), plot_micro=True, plot_macro=True,
                       classes_to_plot=None, title="")
plt.legend(prop={'family': 'Arial', 'weight': 'normal', 'size': 13})
plt.xlabel('False positive rate', fontsize=15, weight='normal', family='Arial')
plt.ylabel('True positive rate', fontsize=15, weight='normal', family='Arial')
plt.xticks(fontsize=15, weight='normal', family='Arial')
plt.yticks(fontsize=15, weight='normal', family='Arial')

plt.savefig('./ROC_curve_DrugBank.png', dpi=1500)
plt.savefig('./ROC_curve_DrugBank.pdf', dpi=1500)

y_pred = []
y_proba = pd.DataFrame(
    pd.read_excel('./y_score.xlsx', sheet_name=0)).values
y = []
for i in range(0, 5, 1):
    a = pd.DataFrame(
        pd.read_excel('./y_fit.xlsx', sheet_name=i)).values
    a = a.flatten()
    b = pd.DataFrame(
        pd.read_excel('./y_score.xlsx', sheet_name=i)).values
    c = pd.read_excel('./y_Test.xlsx', sheet_name=i).values
    c = c.flatten()
    y_pred = np.concatenate([y_pred, a], axis=0)
    if i > 0:
        y_proba = np.concatenate([y_proba, b], axis=0)
    y = np.concatenate([y, c], axis=0)

accuracy = accuracy_score(y, y_pred)
print('Accuracy：%.4f' % accuracy)
n_classes = 7

if n_classes == 2:
    precision = precision_score(y, y_pred)
else:
    precision = precision_score(y, y_pred, average='macro')
print('Precision：%.4f' % precision)

if n_classes == 2:
    recall = recall_score(y, y_pred)
else:
    recall = recall_score(y, y_pred, average='macro')
print('Recall：%.4f' % recall)

if n_classes == 2:
    f1 = f1_score(y, y_pred)
else:
    f1 = f1_score(y, y_pred, average='macro')
print('F1：%.4f' % f1)

actual_labels = set(np.unique(y).tolist())

predicted_labels = set(range(y_proba.shape[1]))

valid_labels = list(actual_labels.intersection(predicted_labels))
valid_labels = [int(label) for label in valid_labels]

y_proba_filtered = y_proba[:, valid_labels]
y_proba_normalized = normalize(y_proba_filtered, norm='l1', axis=1)

if n_classes == 2:
    auc = roc_auc_score(y, y_pred)
else:
    auc = roc_auc_score(y, y_proba, multi_class='ovr', average='macro', labels=valid_labels)
print('AUC：%.4f' % auc)

if n_classes == 2:
    AP = average_precision_score(y, y_pred)
else:
    AP = average_precision_score(y, y_proba, average='macro')
print('AP：%.4f\n' % AP)

y = y.astype(int)
skplt.metrics.plot_roc(y, y_proba, figsize=(6.5, 5.8333), plot_micro=True, plot_macro=True,
                       classes_to_plot=None, title="")
plt.legend(prop={'family': 'Arial', 'weight': 'normal', 'size': 13})
plt.xlabel('False positive rate', fontsize=15, weight='normal', family='Arial')
plt.ylabel('True positive rate', fontsize=15, weight='normal', family='Arial')
plt.xticks(fontsize=15, weight='normal', family='Arial')
plt.yticks(fontsize=15, weight='normal', family='Arial')

plt.savefig('./ROC_curve.png', dpi=1500)
plt.savefig('./ROC_curve.pdf', dpi=1500)
