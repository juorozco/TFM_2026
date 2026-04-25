import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

df_morgan = pd.read_csv("data/db_for_morgan.csv")
df_morgan = df_morgan.dropna(subset=["canonical_smiles"])

# SMILES a molècules RDKit: 
molecules = [Chem.MolFromSmiles(smile) for smile in df_morgan["canonical_smiles"]]

df_morgan["mol_valid"] = [m is not None for m in molecules]

# Molècules RDKit a MorganFingerPrints:
def morganFP(molecules):
    Morgan_FP = []

    for molecule in tqdm(molecules):
        fingerP = AllChem.GetMorganFingerprintAsBitVect(molecule, radius = 2, nBits = 2048)

        fP_arr = np.zeros((2048,), dtype = np.int8)
        DataStructs.ConvertToNumpyArray(fingerP, fP_arr)

        Morgan_FP.append(fP_arr)
        
    return np.array(Morgan_FP)

# S'aplica la funció:
MGFP = morganFP(molecules)

print(len(MGFP))
print(type(MGFP[0]))

np.save("morgan_fingerprints.npy", MGFP)
df_morgan.to_csv("db_with_fingerprints.csv", index=False)