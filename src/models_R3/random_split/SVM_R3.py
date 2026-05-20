import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment

# Càrrega de dades:
X = np.load("data/morgan_fingerprints_R3.npy") # matriu amb els descriptors moleculars
Y = np.load("data/Y_matrix.npy") # matriu amb les dades d'activitat

# Partició:
mol_idx = np.arange(X.shape[0])
train, test = train_test_split(mol_idx, test_size = 0.2, random_state = 42, shuffle = True)

# Mètriques:
targets = Y.shape[1]

roc_auc_scores = []
pr_auc_scores = []
bedroc_scores = []
enrich_factor = []
svm_models = {}

nonvalid_targets = 0

# Model SVM amb radi = 3:
for target in range(targets):

    total_y_train = Y[train, target]
    total_y_test = Y[test, target]

    train_mask = ~np.isnan(total_y_train)
    test_mask  = ~np.isnan(total_y_test)

    X_train = X[train][train_mask]
    y_train = total_y_train[train_mask]

    X_test = X[test][test_mask]
    y_test = total_y_test[test_mask]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2: # les dues classes als dos subconjunts
        nonvalid_targets += 1
        continue

    SVM_model = LinearSVC(C = 1.0, class_weight = "balanced", max_iter = 5000)
    SVM_model.fit(X_train, y_train)

    scores = SVM_model.decision_function(X_test)

    svm_models[target] = SVM_model


    roc_auc_scores.append(roc_auc_score(y_test, scores))
    pr_auc_scores.append(average_precision_score(y_test, scores))

    order = np.argsort(-scores)
    scores_array = np.column_stack([scores[order], y_test[order]])

    bedroc_scores.append(CalcBEDROC(scores_array, col = 1, alpha = 20.0))
    enrich_factor.append(CalcEnrichment(scores_array, col = 1, fractions = [0.01])[0])

# Resultats:
print("Mean ROC-AUC:", np.mean(roc_auc_scores))
print("Mean PR-AUC:", np.mean(pr_auc_scores))
print("Mean BEDROC:", np.nanmean(bedroc_scores))
print("Mean EF@1%:", np.nanmean(enrich_factor))
print("STD ROC-AUC:", np.std(roc_auc_scores))
print("Targets evaluated:", len(roc_auc_scores))