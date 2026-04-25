from chembl_webresource_client.new_client import new_client
import pandas as pd

df_complete = pd.read_csv("data/db_complete.csv")

molecule = new_client.molecule

molecule_id = df_complete['molecule_chembl_id'].unique().tolist()
all_smiles = []

for mol_id in molecule_id:
    mols = molecule.filter(molecule_chembl_id = mol_id).only([
        'molecule_chembl_id',
        'molecule_structures'
        ])
    for mol in mols:
        canonical_smiles = None
        if mol.get('molecule_structures'):
            canonical_smiles = mol['molecule_structures'].get('canonical_smiles')

        all_smiles.append({
            'molecule_chembl_id': mol['molecule_chembl_id'],
            'canonical_smiles': canonical_smiles
        })

# Crear el DataFrame amb tots els resultats
df_smiles = pd.DataFrame(all_smiles)

# Guardar l'arxiu CSV final
df_smiles.to_csv("chembl_smiles.csv", index = False)

# Valors NA:
print(df_smiles['canonical_smiles'].isna().sum())
df_smiles = df_smiles.dropna(subset=['canonical_smiles'])

df_smiles.to_csv("data/partials/chembl_smiles.csv", index = False)

# Unió dels dos llistats obtinguts mitjançant el paràmetre molecule_chembl_id:

df_morgan = df_complete.merge(
    df_smiles,
    on = "molecule_chembl_id",
    how = "left"
)

print(df_morgan.head(5))
print(len(df_morgan))

df_morgan.to_csv("data/db_for_morgan.csv")