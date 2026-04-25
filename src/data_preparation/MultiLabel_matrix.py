import pandas as pd
import numpy as np

df_fingerP = pd.read_csv("db_with_fingerprints.csv") # arxiu final amb les dades
df_fingerP = df_fingerP.drop(columns=["mol_valid"])  # eliminació de la columna que no interessa (mol_valid)

# Canvi de active/inactive per 1/0:
df_fingerP = df_fingerP.replace({
    "active": 1,
    "inactive": 0
})

print(df_fingerP.duplicated().sum()) # 0 duplicats

# Canvi de forma per crear la matriu multi-label
df_matrix = pd.crosstab(
    df_fingerP["molecule_chembl_id"],
    df_fingerP["target_chembl_id"],
    values=df_fingerP["activity_type"],
    aggfunc="max"
)

df_matrix.to_csv("df_matrix.csv")

print(len(df_matrix)) # la taula girada te 169133 files, que és el total de molècules del dataset
print(df_fingerP["molecule_chembl_id"].nunique()) # mateix nombre de molècules, correcte.

X = np.load("morgan_fingerprints.npy")
print(X.shape) # s'ha de canviar perque les files corresponguin al total de molècules del dataset igual que df_matrix

print(df_matrix.isna().sum().sum())
total_values = df_matrix.shape[0] * df_matrix.shape[1]
nan_values = df_matrix.isna().sum().sum()

print("Sparsity (% NaN):", round(nan_values / total_values * 100, 2))

counts = df_matrix.sum(axis=0)

print((counts < 50).sum())