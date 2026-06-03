import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

# Obtenció dels Morgan fingerprints de radi = 3:

df_smiles = pd.read_csv("data/chembl_smiles.csv")
df_smiles = df_smiles.dropna(subset=["canonical_smiles"])

# SMILES a molècules RDKit: 
molecules = [Chem.MolFromSmiles(smile) for smile in df_smiles["canonical_smiles"]]
df_smiles["mol_valid"] = [m is not None for m in molecules]

# Funció per a l'obtenció dels Morgan fingerprints:
def morganFP(molecules):
    Morgan_FP = []

    for molecule in tqdm(molecules):
        fingerP = AllChem.GetMorganFingerprintAsBitVect(molecule, radius = 3, nBits = 2048)

        fP_arr = np.zeros((2048,), dtype = np.int8)
        DataStructs.ConvertToNumpyArray(fingerP, fP_arr)

        Morgan_FP.append(fP_arr)
        
    return np.array(Morgan_FP)

# S'aplica la funció al llistat de molècules RDKit:
MGFP = morganFP(molecules)

# Eliminació de la columna mol_valid
df_smiles = df_smiles.drop(columns=["mol_valid"])  
np.save("data/morgan_fingerprints_R3.npy", MGFP)