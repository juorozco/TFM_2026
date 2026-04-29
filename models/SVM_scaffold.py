import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X = np.load("data/morgan_fingerprints.npy") # matriu amb els descriptors moleculars
Y = np.load("data/Y_matrix.npy") # matriu amb els descriptors moleculars
df_smiles = pd.read_csv("data/chembl_smiles.csv") # arxiu amb els SMILES

# Descàrrega dels scaffolds:
def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol = mol)
    except:
        return None

df_smiles["scaffold"] = df_smiles["canonical_smiles"].apply(get_scaffold)

mask = df_smiles["scaffold"].notnull()
df_smiles = df_smiles[mask].reset_index(drop = True)
X = X[mask.values]
Y = Y[mask.values]

print(len(df_smiles))
print(df_smiles["scaffold"].nunique())

# Scaffold split en 70/30:
scaffolds = df_smiles["scaffold"].unique()
train_scaffold, test_scaffold = train_test_split(scaffolds, test_size = 0.3, random_state = 42, shuffle = True)

train_idx = df_smiles.index[df_smiles["scaffold"].isin(train_scaffold)].values
test_idx  = df_smiles.index[df_smiles["scaffold"].isin(test_scaffold)].values

targets = Y.shape[1]

auc_values = []
auc_train_values = []
svm_models = {}
nonvalid_targets = 0

# Model LinearSVC amb scaffold split:
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

    SVM_scaffold_model = LinearSVC(C = 1.0, class_weight = "balanced", max_iter = 5000)
    SVM_scaffold_model.fit(X_train, y_train)

    auc_values.append(roc_auc_score(y_test, SVM_scaffold_model.decision_function(X_test)))
    auc_train_values.append(roc_auc_score(y_train, SVM_scaffold_model.decision_function(X_train)))

    svm_models[target] = SVM_scaffold_model

# Resultats:
print("Mitjana ROC-AUC test =", np.mean(auc_values)) # 0.79
print("Coverage =", len(auc_values) / targets * 100, "%") # 54.72%
print("GAP =", np.mean(auc_train_values) - np.mean(auc_values)) # 0.21
print("Percentils test valors AUC:", np.percentile(auc_values, [0, 25, 50, 75, 100])) # [0.17 0.74 0.80 0.86 0.98]