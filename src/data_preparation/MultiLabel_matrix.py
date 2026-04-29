import pandas as pd
import numpy as np

df_smiles = pd.read_csv("data/chembl_smiles.csv") # arxiu amb les molècules que tenen smiles vàlids
db_complete = pd.read_csv("data/db_complete.csv") # arxiu final amb les dades

valid_mols = set(df_smiles["molecule_chembl_id"]) # detecció de les molècules vàlides perquè el total no coincideix
db_complete = db_complete[db_complete["molecule_chembl_id"].isin(valid_mols)] # agafa només les que estan a df_smiles i són vàlides

# Canvi de active/inactive per 1/0:
db_complete = db_complete.replace({
    "active": 1,
    "inactive": 0
})

print(db_complete.duplicated().sum()) # 0 duplicats

# Canvi de forma (long to wide) per crear la matriu multi-label
df_matrix = pd.crosstab(
    db_complete["molecule_chembl_id"],
    db_complete["target_chembl_id"],
    values=db_complete["activity_type"],
    aggfunc="max"
)

df_matrix.to_csv("data/df_matrix.csv")

print(len(df_matrix)) # la taula girada te 169133 files, que és el total de molècules del dataset
print(db_complete["molecule_chembl_id"].nunique()) # mateix nombre de molècules, correcte.

# El dataset és sparse:
print(df_matrix.isna().sum().sum())
total_values = df_matrix.shape[0] * df_matrix.shape[1]
nan_values = df_matrix.isna().sum().sum()
print("Sparsity (% NaN):", round(nan_values / total_values * 100, 2))

mol_order = df_smiles["molecule_chembl_id"].values
df_matrix = df_matrix.reindex(mol_order)
Y = df_matrix.values.astype(np.float32)
mol_order = df_matrix.index.values

np.save("data/Y_matrix.npy", Y)
print("✔ Y shape:", Y.shape)

# Preparació de X per a la matriu:
X = np.load("data/morgan_fingerprints.npy")
print(X.shape) # les files han de correspondre al total de molècules del dataset igual que df_matrix

# Ordre de les molècules (ha de ser el mateix):

assert X.shape[0] == Y.shape[0]

print("✔ X shape:", X.shape)
print("✔ Y shape:", Y.shape)
print("✔ ALIGNMENT OK")