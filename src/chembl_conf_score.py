from chembl_webresource_client.new_client import new_client
import pandas as pd

# Obtenció de l'assay_type i el confidence_score per a cada assay_ID, per completar la base de dades:

df = pd.read_csv("data/db_final.csv") # càrrega de l'arxiu amb compostos, activitats i assays ID obtingut anteriorment

assay = new_client.assay
conf_scores = []

for i, target_id in df['target_chembl_id'].unique():
    conf_score = assay.filter(
        target_chembl_id = target_id
    ).only([
        'assay_chembl_id',
        'confidence_score',
        'assay_type'
    ])
    for conf in conf_score:
        conf_scores.append({
            "target_chembl_id": target_id,
            "assay_chembl_id": conf["assay_chembl_id"],
            "confidence_score": conf["confidence_score"],
            "assay_type": conf["assay_type"]
        })

df_conf_assays = pd.DataFrame(conf_scores)
df_conf_assays.to_csv("conf_assays_types.csv", index = False)













 

