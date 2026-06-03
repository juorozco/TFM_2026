import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from chembl_webresource_client.new_client import new_client

# Càrrega de les dades i definició del model FNN:
X = np.load("data/morgan_fingerprints.npy") # fingerprints moleculars (Morgan fingerprints)
Y = np.load("data/Y_matrix.npy") # matriu multi-label d'activitat

df_matrix = pd.read_csv("data/partials/df_matrix.csv", index_col = 0)
df_kinases = pd.read_csv("data/biological_analysis/kinase_list.csv") # llistat amb el nom de les cinases

kinase_names = df_matrix.columns.tolist()
molecule_names = df_matrix.index.tolist()

# Model FNN:
mol_idx = np.arange(X.shape[0])
train_val_idx, test_idx = train_test_split(mol_idx, test_size = 0.20, random_state = 42)

device = "cuda" if torch.cuda.is_available() else "cpu"

class FeedForward_NN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

FNN_model = FeedForward_NN(2048, 318).to(device)
FNN_model.load_state_dict(torch.load("src/best_models/best_FNN_model.pt", map_location = device))
FNN_model.eval()

# Prediccions sobre el conjunt de test:
X_test = torch.from_numpy(X[test_idx].astype(np.float32)).to(device)

# Gradient:
with torch.no_grad():
    preds = torch.sigmoid(FNN_model(X_test)).cpu().numpy()

# Llindar de probabilitat = 0.5:
preds_binary = (preds >= 0.5).astype(int)

# Els 20 compostos amb més interaccions off-target predites:
n_targets_per_compound = preds_binary.sum(axis = 1)
top20_idx = np.argsort(-n_targets_per_compound)[:20]
top20_ids = [molecule_names[test_idx[i]] for i in top20_idx]
top20_molecules = n_targets_per_compound[top20_idx]
print(top20_ids, top20_molecules)

# Les 20 cinases amb més interaccions predites:
n_actives_per_kinase = preds_binary.sum(axis = 0)
df_kinases = pd.DataFrame({'target_chembl_id': kinase_names, 'n_actives_pred': n_actives_per_kinase})

df_kinases = df_kinases.merge(df_kinases, on = 'target_chembl_id')
df_kinases = df_kinases.sort_values('n_actives_pred', ascending = False)

top20_kinases = df_kinases.head(20)

plt.figure(figsize = (14, 6))
plt.barh(top20_kinases['pref_name'][::-1], top20_kinases['n_actives_pred'][::-1], color = 'xkcd:grey green', 
         edgecolor = 'white', height = 1.0)
plt.xlabel('Nombre de compostos predits com a actius')
plt.tight_layout()
plt.savefig("figures/top20_cinases.png", dpi = 300)
plt.show()

# Heatmap dels 20 compostos amb les 20 cinases amb més interaccions predites:
top20_kinase_idx = [kinase_names.index(k) for k in df_kinases['target_chembl_id'].head(20).tolist()]
heatmap_data = preds[top20_idx][:, top20_kinase_idx]
kinase_short_names = df_kinases['pref_name'].head(20).tolist()

plt.figure(figsize = (14, 6))
sns.heatmap(heatmap_data, xticklabels = kinase_short_names, yticklabels = top20_ids, cmap = 'copper', vmin = 0,
    vmax = 1, linewidths = 0.3, annot = False)
plt.xlabel('Cinases')
plt.ylabel('ChEMBL ID compostos')
plt.xticks(rotation = 45, ha = 'right', fontsize = 8)
plt.yticks(fontsize = 8)
plt.tight_layout()
plt.savefig("figures/heatmap.png", dpi = 300)
plt.show()

# Cinases amb més interaccions segons les dades reals del conjunt de dades:
df_matrix    = pd.read_csv("data/partials/df_matrix.csv", index_col = 0)
n_actives_real = (df_matrix == 1).sum(axis = 0).sort_values(ascending = False)

df_real = pd.DataFrame({'target_chembl_id': n_actives_real.index, 'n_actives_real': n_actives_real.values})
df_real = df_real.merge(df_kinases, on = 'target_chembl_id')

print(df_real[['pref_name', 'n_actives_real']].head(20).to_string())

# Validació mitjançant ChEMBL:
activity = new_client.activity
results = []

for i, (chembl_id, pred_idx) in enumerate(zip(top20_ids, top20_idx)):

    acts = activity.filter(molecule_chembl_id = chembl_id, pchembl_value__isnull = False).only(['target_chembl_id', 'pchembl_value', 'standard_type'])
    
    # Filtre per cinases al llistat + eliminació de dades sense pChEMBL vàlid:
    df_acts = pd.DataFrame(list(acts))
    df_acts['pchembl_value'] = pd.to_numeric(df_acts['pchembl_value'], errors = 'coerce')
    
    # Diana principal = cinasa amb pChEMBL més alt:
    df_acts = df_acts.dropna(subset = ['pchembl_value'])
    df_acts = df_acts[df_acts['target_chembl_id'].isin(kinase_names)]
    main_target = df_acts.loc[df_acts['pchembl_value'].idxmax()]
    main_target_id = main_target['target_chembl_id']
    main_target_pchembl = main_target['pchembl_value']
    
    # Probabilitat predita pel model per a la diana principal:
    main_target_idx = kinase_names.index(main_target_id)
    main_target_prob = preds[pred_idx, main_target_idx]
    
    # Posició al rànquing de la diana principal:
    compound_preds = preds[pred_idx]
    order = np.argsort(-compound_preds)
    rank = np.where(order == main_target_idx)[0][0] + 1
    
    # Nom de la diana principal:
    main_target_name = df_kinases[df_kinases['target_chembl_id'] == main_target_id]['pref_name'].values
    main_target_name = main_target_name[0] if len(main_target_name) > 0 else main_target_id

    results.append({
        'compound_chembl_id': chembl_id,
        'main_target_id': main_target_id,
        'main_target_name': main_target_name,
        'pchembl_value': main_target_pchembl,
        'predicted_prob': round(main_target_prob, 4),
        'rank': rank})

# Resultats:
df_results = pd.DataFrame(results)
print(df_results.to_string())