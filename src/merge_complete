from chembl_webresource_client.new_client import new_client
import pandas as pd

# Unió dels dos llistats obtinguts mitjançant el paràmetre assay_ID:

df_acts_assays = pd.read_csv("data/db_final.csv")
df_conf_assays = pd.read_csv("data/conf_assays_types.csv")

df_complete = df_acts_assays.merge(
    df_conf_assays,
    on = "assay_chembl_id",
    how = "left"
)

df_complete.to_csv("data/db_complete.csv", index = False)
print(len(df_complete))

# Comprovació de que l'arxiu final no presenta ni valors NA ni duplicats:
print(len(df_acts_assays), len(df_complete)) # mateix nombre de registres entre els arxius
print(df_complete[['confidence_score', 'assay_type']].isna().sum()) # sense valors NA
print(df_complete.duplicated().sum()) # sense duplicats