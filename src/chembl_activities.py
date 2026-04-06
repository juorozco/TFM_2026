from chembl_webresource_client.new_client import new_client
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/db.tsv", sep='\t')  # carrega de les dades extretes de la web de ChEMBL 
print(data.head())

data = data.drop(columns=['Species Group Flag', 'Tax ID']) # Neteja de les dues columnes que no faré servir
print(data.head())
print(len(data))

# Filtre per targets que tinguin almenys 50 compostos associats:
data_compounds = data[data['Compounds'] > 50]
data_compounds = data_compounds.sort_values(by = 'Compounds', ascending = True) # ordre ascendent respecte columna compounds
print(len(data_compounds)) # comprobació de que el nombre s'ha reduït

# Boxplot d'outliers per eliminar els compostos amb molt poques activitats:
plt.figure()
data_compounds['Activities'].plot(kind='box')
plt.title("Distribució d'activitats per cinasa")
plt.ylabel("Nombre d'activitats")
plt.show() # El boxplot mostra que les cinases amb menys activitats associades són les que tenen menys de 700.

# Es decideix eliminar només els targets amb menys de 700 activitats. 
# Els que tenen moltes activitats associades són targets importants i, encara que siguin outliers, es conserven.
df_filtered = data_compounds[data_compounds['Activities'] > 700]
print(len(df_filtered)) # Queden 318 cinases

df_filtered = df_filtered.sort_values(by = 'Activities', ascending = True)
print(df_filtered.head())

# Associació dels targets amb els seus compostos associats i les seves activitats:
activity = new_client.activity

kinases = df_filtered['ChEMBL ID'].tolist()
all_activities = []
total = len(kinases)

for i, target_id in enumerate(kinases, start = 1):
    print(f"Target {i}/{total}: {target_id}")

    act = activity.filter(target_chembl_id = target_id).only([
        'molecule_chembl_id',
        'target_chembl_id',
        'pchembl_value',
        'standard_value',
        'standard_type'
    ])

    for a in act:
        if a.get('pchembl_value') is not None:
            all_activities.append(a)

df_acts = pd.DataFrame(all_activities)
df_acts.to_csv("data/total_activities.csv", index=False)

