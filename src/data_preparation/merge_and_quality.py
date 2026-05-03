from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_acts_assays = pd.read_csv("data/partials/db_final.csv")
df_conf_assays = pd.read_csv("data/partials/conf_assays_types.csv")

print(len(df_acts_assays)) # 525450 registres totals

# Unió dels dos llistats obtinguts mitjançant el paràmetre assay_ID:
df_acts = df_acts_assays.merge(
    df_conf_assays,
    on = "assay_chembl_id",
    how = "left"
)

df_acts['target_chembl_id'] = df_acts['target_chembl_id_x']
df_acts = df_acts.drop(columns=['target_chembl_id_x', 'target_chembl_id_y'])

# Filtres:
# confidence_score: només registres que tinguin un valor de confiança del 9
# assay_type: només assajos de binding i functional, els millors per als models de classificació

df_acts = df_acts[(df_acts['confidence_score'] == 9) &
                          (df_acts['assay_type'].isin(['B','F']))]

print(len(df_acts)) # 344740
print(df_acts['confidence_score'].value_counts(dropna=False)) #344740
print(df_acts['assay_type'].value_counts(dropna=False)) # només B i F

# Calcul de la mediana del valor pChEMBL per gestionar els duplicats:

df_complete = df_acts.groupby(['molecule_chembl_id', 'target_chembl_id'], 
                                     as_index = False)['pchembl_value'].median()
print(len(df_complete)) # 245614 despres de la mediana

# Distribució de les dades:
print("Mitjana pChEMBL:", round(df_complete['pchembl_value'].mean(), 4))
print("STD pChEMBL:", round(df_complete['pchembl_value'].std(), 4))

# Histograma pChEMBL:
plt.figure(figsize = (8, 5))
plt.hist(df_complete['pchembl_value'], bins = 50, color = 'salmon', edgecolor = 'whitesmoke')
plt.axvline(x = 6.5, color = 'red', linestyle = '--', linewidth = 1.5, label = 'Llindar activitat (pChEMBL = 6.5)')
plt.xlabel('pChEMBL')
plt.ylabel('Nombre de registres')
plt.legend()
plt.tight_layout()
plt.savefig("figures/distribució_pchembl.png", dpi = 300)
plt.show()

# Classificació actius/inactius segons valor pChEMBL:
df_complete['activity_type'] = np.where(
    df_complete['pchembl_value'] >= 6.5,
    'active',
    'inactive'
)

# Checks:
print(df_complete["pchembl_value"].isna().sum()) # 0 valors Na

# Percentatge de molècules actives vs inactives en tant per cent:
total = len(df_complete)
active = np.sum(df_complete['activity_type'] == 'active')
inactive = np.sum(df_complete['activity_type'] == 'inactive')

print("Actius:", round(active / total * 100, 2), "%") # 63.45%
print("Inactius:", round(inactive / total * 100, 2), "%") # 36.55%

df_complete.to_csv("data/db_complete.csv", index = False)
print(len(df_complete))