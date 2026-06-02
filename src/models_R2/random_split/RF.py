import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment

# Càrrega de dades:
X = np.load("data/morgan_fingerprints.npy") # fingerprints moleculars (Morgan fingerprints)
Y = np.load("data/Y_matrix.npy") # matriu multi-label d'activitat

# Random split (train 80%, test 20%) i reproduïbilitat:
mol_idx = np.arange(X.shape[0])
train, test = train_test_split(mol_idx, test_size = 0.20, random_state = 42, shuffle = True)

# Inicialització de les variables d'avaluació:
targets = Y.shape[1]

roc_auc_scores = []
pr_auc_scores = []
bedroc_scores = []
enrich_factor = []

# Diccionari per emmagatzemar un model per cada cinasa:
rf_models = {}

# Targets descartats:
nonvalid_targets = 0

# RF:
for target in range(targets):
    
    # Separació de labels per cinasa
    total_y_train = Y[train, target]
    total_y_test = Y[test, target]
    
    # Eliminació de valors NaN
    train_mask = ~np.isnan(total_y_train)
    test_mask  = ~np.isnan(total_y_test)

    X_train = X[train][train_mask]
    y_train = total_y_train[train_mask]
    X_test = X[test][test_mask]
    y_test = total_y_test[test_mask]

    # Filtre per assegurar les dues classes
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        nonvalid_targets += 1
        continue

    RF_model = RandomForestClassifier(n_estimators = 100, class_weight = 'balanced', random_state = 42, n_jobs =- 1)
    RF_model.fit(X_train, y_train)
    
    # Probabilitat de classe positiva
    scores = RF_model.predict_proba(X_test)[:, 1]

    rf_models[target] = RF_model

    # Avaluació:
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
