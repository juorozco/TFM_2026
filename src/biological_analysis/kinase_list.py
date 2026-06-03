from chembl_webresource_client.new_client import new_client
import pandas as pd

# Llista dels IDs de ChEMBL de les cinases:
df_matrix = pd.read_csv("data/partials/df_matrix.csv", index_col = 0)
kinase_ids = df_matrix.columns.tolist()

# Obtenció dels noms de les cinases via ChEMBL:
target = new_client.target
kinases_list = []

for id in kinase_ids:
    result = target.get(id)
    kinases_list.append({'target_chembl_id': id, 'pref_name': result['pref_name']})

df_kinases = pd.DataFrame(kinases_list)
df_kinases.to_csv("data/biological_analysis/kinase_list.csv", index = False)
print(df_kinases.head())