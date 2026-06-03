from chembl_webresource_client.new_client import new_client
import pandas as pd
import matplotlib.pyplot as plt

# Càrrega de les dades extretes de ChEMBL:
data = pd.read_csv("data/partials/db.tsv", sep = '\t')
print(data.head())

# Neteja de les dues columnes que no es necessitaran per a l'estudi:
data = data.drop(columns = ['Species Group Flag', 'Tax ID'])
print(len(data))

# Filtre per targets que tinguin almenys 50 compostos associats:
data_compounds = data[data['Compounds'] > 50]
print(len(data_compounds))

# Boxplot d'activitats:
plt.figure(figsize = (6,6))
plt.boxplot(
    data_compounds['Activities'],
    patch_artist = True,
    boxprops = dict(facecolor = 'xkcd:grey green'),
    medianprops = dict(color = 'xkcd:pine', linewidth = 1.5))
plt.axhline(700, linestyle = '--', linewidth = 1, color = 'xkcd:eggplant')
plt.ylabel("Nombre d'activitats")
plt.tight_layout()
plt.savefig("figures/Boxplot_activitats.png", dpi = 300)
plt.show()

# Eliminació de dianes amb < 700 activitats:
df_filtered = data_compounds[data_compounds['Activities'] > 700]
print(len(df_filtered))
df_filtered.to_csv("data/partials/db_filtered.csv", index = False)

# Associació de les cinases amb els compostos, les activitats i els assays corresponents:
activity = new_client.activity

kinases = df_filtered['ChEMBL ID'].tolist()
all_activities = []
total = len(kinases)

for target_id in kinases:
    acts = activity.filter(target_chembl_id = target_id).only([
        'molecule_chembl_id',
        'target_chembl_id',
        'pchembl_value',
        'standard_value',
        'standard_type',
        'assay_chembl_id'
    ])
    for act in acts:
        if act.get('pchembl_value') is not None:
            all_activities.append(act)

df_acts = pd.DataFrame(all_activities)
df_acts.to_csv("data/partials/db_final.csv", index = False)