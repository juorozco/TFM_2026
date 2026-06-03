import pandas as pd
import numpy as np

# Obtenció de la matriu multi-label per a l'aplicació dels models.

df_smiles = pd.read_csv("data/chembl_smiles.csv") # SMILES de les molècules
db_complete = pd.read_csv("data/db_complete.csv") # Dades d'activitat finals

# Molècules amb SMILES vàlid:
valid_molecules = set(df_smiles["molecule_chembl_id"]) 
db_complete = db_complete[db_complete["molecule_chembl_id"].isin(valid_molecules)] 

# Binarització de l'activitat: actius = 1 / inactius = 0.
db_complete = db_complete.replace({"active": 1, "inactive": 0})

# Comprovació de que no hi hagi duplicats:
print(db_complete.duplicated().sum())

# Matriu multi-label: les files són les molècules i les columnes són les cinases:
df_matrix = pd.crosstab(db_complete["molecule_chembl_id"], db_complete["target_chembl_id"], 
                        values = db_complete["activity_type"], aggfunc = "max")

df_matrix.to_csv("data/df_matrix.csv")

# Comprovació de que el total de molècules úniques correspon amb el total de files de la matriu:
print(len(df_matrix))
print(db_complete["molecule_chembl_id"].nunique())

# Esparsitat del dataset:
nan_values = df_matrix.isna().sum().sum()
total_values = df_matrix.size
print("% Sparsity:", round(nan_values / total_values * 100, 2))

# Creació i reordenació de la matriu d'activitat (Y) perquè coincideixi amb el llistat de Morgan fingerprints:
molecules_order = df_smiles["molecule_chembl_id"].values
df_matrix = df_matrix.reindex(molecules_order)
Y = df_matrix.values.astype(np.float32)
molecules_order = df_matrix.index.values

np.save("data/Y_matrix.npy", Y)
print(Y.shape)

# Preparació de X (Morgan fingerprints) per a l'aplicació dels models:
X = np.load("data/morgan_fingerprints.npy")
print(X.shape)

# X i Y han de tenir el mateix nombre de files. Cada fila representa la mateixa molècula en ambdues matrius:
assert X.shape[0] == Y.shape[0]