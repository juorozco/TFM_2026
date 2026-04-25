from chembl_webresource_client.new_client import new_client
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/db.tsv", sep = '\t')  # carrega de les dades extretes de la web de ChEMBL 
print(data.head())

data = data.drop(columns = ['Species Group Flag', 'Tax ID']) # Neteja de les dues columnes que no es necessiten
print(len(data))

# Filtre per targets que tinguin almenys 50 compostos associats:
data_compounds = data[data['Compounds'] > 50]
print(len(data_compounds)) # comprobació de que el nombre s'ha reduït

# Boxplot d'outliers per eliminar els compostos amb molt poques activitats:
plt.figure()
data_compounds['Activities'].plot(kind = 'box')
plt.title("Activities")
plt.ylabel("Activities")
plt.show() # El boxplot mostra que les cinases amb menys activitats associades són les que tenen menys de 700.
plt.savefig("figures/Boxplot_acts_filter.png")

# Es decideix eliminar els targets amb menys de 700 activitats. 
# Els que tenen moltes activitats associades són targets importants i, encara que siguin outliers, es conserven.

df_filtered = data_compounds[data_compounds['Activities'] > 700]
print(len(df_filtered)) # Queden 318 cinases

df_filtered.to_csv("data/partials/db_filtered.csv", index = False)

# Associació dels targets amb els compostos, les activitats i els assays corresponents:
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

print(len(df_acts))
#