from chembl_webresource_client.new_client import new_client
import pandas as pd

db = pd.read_csv("data/db_filtered.csv")
print(len(db))

activity = new_client.activity
kinases = db['ChEMBL ID'].tolist()  # 318 targets

partial_file = "assay_acts_partial.csv"

# Inicialitzar fitxer parcial (sobreescriu si existeix)
with open(partial_file, 'w') as f:
    pass

all_data = []

for i, target_id in enumerate(kinases, start=1):
    print(f"Target {i}/{len(kinases)}: {target_id}")

    acts = activity.filter(
        target_chembl_id=target_id
    ).only([
        'molecule_chembl_id',
        'target_chembl_id',
        'assay_chembl_id',
        'pchembl_value'
    ])

    for a in acts:
        if a.get('pchembl_value') is not None:
            all_data.append(a)

    # 🔹 Guardar després de cada target
    df_partial = pd.DataFrame(all_data)
    df_partial.to_csv(partial_file, index=False)
    print(f"Guardat checkpoint al target {i}")

# 🔹 Guardar tot al final també
df_assay_acts = pd.DataFrame(all_data)
df_assay_acts.to_csv("assay_acts.csv", index=False)
print(len(df_assay_acts))


# NO INCLOURE