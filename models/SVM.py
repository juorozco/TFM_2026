import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X = np.load("data/morgan_fingerprints.npy") # matriu amb els descriptors moleculars
Y = np.load("data/Y_matrix.npy") # matriu amb les dades d'activitat

# Random split en 70/30:
mol_idx = np.arange(X.shape[0])
train_idx, test_idx = train_test_split(mol_idx, test_size = 0.3, random_state = 42, shuffle = True)

targets = Y.shape[1]

auc_values = []
auc_train_values = []
svm_models = {}
nonvalid_targets = 0

# Model LinearSVC:
for target in range(targets):
    
    total_y_train = Y[train_idx, target]
    total_y_test = Y[test_idx, target]

    train_mask = ~np.isnan(total_y_train)
    test_mask  = ~np.isnan(total_y_test)

    X_train = X[train_idx][train_mask]
    y_train = total_y_train[train_mask]

    X_test = X[test_idx][test_mask]
    y_test = total_y_test[test_mask]

    if len(y_train) < 100:  # mínim de 100 observacions
        nonvalid_targets += 1
        continue

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2: # les dues classes als dos subconjunts
        nonvalid_targets += 1
        continue

    SVM_model = LinearSVC(C = 1.0, class_weight = "balanced", max_iter = 5000)
    SVM_model.fit(X_train, y_train)

    auc_values.append(roc_auc_score(y_test, SVM_model.decision_function(X_test)))
    auc_train_values.append(roc_auc_score(y_train, SVM_model.decision_function(X_train)))

    svm_models[target] = SVM_model

# Resultats:
print("Mitjana ROC-AUC test =", np.mean(auc_values)) # 0.84
print("Coverage =", len(auc_values) / targets * 100, "%") # 55.35%
print("GAP =", np.mean(auc_train_values) - np.mean(auc_values)) # 0.16
print("Percentils test valors AUC:", np.percentile(auc_values, [0, 25, 50, 75, 100])) # [0.52 0.80 0.85 0.89 0.98]
