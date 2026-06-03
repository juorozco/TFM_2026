from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Unió dels dos llistats mitjançant el paràmetre assay_ID per a la posterior obtenció de la matriu multi-label.

df_acts_assays = pd.read_csv("data/partials/db_final.csv")
df_conf_assays = pd.read_csv("data/partials/conf_assays_types.csv")

df_acts = df_acts_assays.merge(df_conf_assays, on = "assay_chembl_id", how = "left")

# Reajust de columnes:
df_acts['target_chembl_id'] = df_acts['target_chembl_id_x']
df_acts = df_acts.drop(columns=['target_chembl_id_x', 'target_chembl_id_y'])

# Aplicació dels filtres de qualitat conf_score = 9 i assay_type = B i F:
df_acts = df_acts[(df_acts['confidence_score'] == 9) & (df_acts['assay_type'].isin(['B','F']))]

print(df_acts['confidence_score'].value_counts(dropna = False))
print(df_acts['assay_type'].value_counts(dropna = False))

# Calcul de la mediana del valor pChEMBL per gestionar els duplicats:
df_complete = df_acts.groupby(['molecule_chembl_id', 'target_chembl_id'], as_index = False)['pchembl_value'].median()
print(len(df_complete))

# Distribució de les dades:
print("Mitjana pChEMBL:", round(df_complete['pchembl_value'].mean(), 4))
print("STD pChEMBL:", round(df_complete['pchembl_value'].std(), 4))

# Histograma de distribució de pChEMBL:
plt.figure(figsize = (8, 5))
plt.hist(df_complete['pchembl_value'], bins = 50, color = 'xkcd:grey green', edgecolor = 'xkcd:white')
plt.axvline(x = 6.5, color = 'xkcd:eggplant', linestyle = '--', linewidth = 1.5, label = 'Llindar activitat (pChEMBL = 6.5)')
plt.xlabel('pChEMBL')
plt.ylabel('Nombre de registres')
plt.legend()
plt.tight_layout()
plt.savefig("figures/distribució_pchembl.png", dpi = 300)
plt.show()

# Classificació molècules actives/inactives segons valor pChEMBL 6.5:
df_complete['activity_type'] = np.where(df_complete['pchembl_value'] >= 6.5, 'active', 'inactive')

# Percentatge de molècules actives vs inactives en tant per cent:
total = len(df_complete)
active = np.sum(df_complete['activity_type'] == 'active')
inactive = np.sum(df_complete['activity_type'] == 'inactive')

print("Actius:", round(active / total * 100, 2), "%")
print("Inactius:", round(inactive / total * 100, 2), "%")

df_complete.to_csv("data/db_complete.csv", index = False)