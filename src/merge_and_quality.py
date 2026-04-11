from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np

# Unió dels dos llistats obtinguts mitjançant el paràmetre assay_ID:

df_acts_assays = pd.read_csv("data/db_final.csv")
df_conf_assays = pd.read_csv("data/conf_assays_types.csv")

# Calcul de la mediana del valor pChEMBL per gestionar els duplicats:

print(len(df_acts_assays)) # 525450 registres totals

dup = ['molecule_chembl_id', 'target_chembl_id', 'assay_chembl_id']
duplicates_acts_assays = df_acts_assays.duplicated(subset = dup, keep = False).sum()
print(duplicates_acts_assays) # 36388 duplicats

df_acts_median = df_acts_assays.groupby(dup, as_index = False)['pchembl_value'].median()
print(len(df_acts_median)) # 506142 despres de la mediana

df_complete = df_acts_median.merge(
    df_conf_assays,
    on = "assay_chembl_id",
    how = "left"
)

# Comprovació de que l'arxiu final no presenta ni valors NA ni duplicats:
print(df_complete[['confidence_score', 'assay_type']].isna().sum()) # sense valors NA
print(df_complete.duplicated().sum()) # sense duplicats

# Eliminació d'una de les dues columnes amb el nom del target (iguals) i canvi de nom de la que queda:
df_complete = df_complete.drop(columns = ['target_chembl_id_y'])
df_complete = df_complete.rename(columns={'target_chembl_id_x': 'target_chembl_id'})
print(df_complete.columns)

# Tipus de confidence_score i assay_type:
counts = df_complete['confidence_score'].value_counts().sort_index()
counts_assay = df_complete['assay_type'].value_counts().sort_index()

print(counts)
print(counts_assay)

# Filtres:
# confidence_score: només registres que tinguin un valor de confiança del 9
# assay_type: només assajos de binding i functional, els millors per als models de classificació

df_complete = df_complete[(df_complete['confidence_score'] == 9) &
                          (df_complete['assay_type'].isin(['B','F']))]


# Continuació post-PAC2:

# classificació actius/inactius segons valor pChEMBL:
df_complete['activity_type'] = np.where(
    df_complete['pchembl_value'] >= 6.5,
    'active',
    'inactive'
)

df_complete.to_csv("data/db_complete.csv", index = False)
print(len(df_complete))

# Percentatge de molècules actives vs inactives en tant per cent:
total = len(df_complete)
active = np.sum(df_complete['activity_type'] == 'active')
inactive = np.sum(df_complete['activity_type'] == 'inactive')

print("Actius:", round(active / total * 100, 2), "%") # 65.82%
print("Inactius:", round(inactive / total * 100, 2), "%") # 34.18%
