import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split
RDLogger.DisableLog('rdApp.*')

# Obtenció de l'scaffold de Murcko de les molècules a partir del seu SMILES:
def get_scaffold(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize = True)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol = mol)
    except:
        return None

# Divisió del conjunt de dades agrupant les molècules pel seu scaffold:
def scaffold_split(arxiu_smiles, val_size = None, test_size = 0.2, random_state = 42):

    df = pd.read_csv(arxiu_smiles)

    df["scaffold"] = df["canonical_smiles"].apply(get_scaffold)

    mask = df["scaffold"].notnull().values
    df = df.loc[mask].reset_index(drop = True)

    scaffolds = df["scaffold"].unique()
    train_val_scaff, test_scaff = train_test_split(scaffolds, test_size = test_size, random_state = random_state, shuffle = True)
    test = df.index[df["scaffold"].isin(test_scaff)].values

    # SVM i RF (no tenen conjunt de validació):
    if val_size is None:
        train = df.index[df["scaffold"].isin(train_val_scaff)].values
        return train, test
    
    # FNN (amb conjunt de validació):
    # Ajust del conjunt de validació perquè la proporció dels tres models sigui la mateixa
    val_size_rel = val_size / (1 - test_size)
    train_scaff, val_scaff = train_test_split(train_val_scaff, test_size = val_size_rel, random_state = random_state, shuffle = True)

    train = df.index[df["scaffold"].isin(train_scaff)].values
    val   = df.index[df["scaffold"].isin(val_scaff)].values

    return train, val, test