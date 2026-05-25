import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Càrrega de les dades i de la FNN:
X = np.load("data/morgan_fingerprints.npy")
Y = np.load("data/Y_matrix.npy")

df_matrix = pd.read_csv("data/partials/df_matrix.csv", index_col = 0)
df_kinases = pd.read_csv("data/kinase_list.csv")

kinase_names = df_matrix.columns.tolist()
molecule_names = df_matrix.index.tolist()

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

X_test = torch.from_numpy(X[test_idx].astype(np.float32)).to(device)
with torch.no_grad():
    preds = torch.sigmoid(FNN_model(X_test)).cpu().numpy()

threshold = 0.5
preds_binary = (preds >= threshold).astype(int)

# Top 20 compostos amb més interaccions off-target:
n_targets_per_compound = preds_binary.sum(axis = 1)
top20_idx = np.argsort(-n_targets_per_compound)[:20]
top20_ids = [molecule_names[test_idx[i]] for i in top20_idx]
top20_molecules = n_targets_per_compound[top20_idx]

plt.figure(figsize = (14, 6))
plt.barh(top20_ids[::-1], top20_molecules[::-1], color = 'xkcd:grey green', edgecolor = 'white', height = 1.0)
plt.xlabel('Nombre de cinases predites com a actives')
plt.title(f'Top 20 compostos amb més interaccions off-target predites (threshold={threshold})')
plt.tight_layout()
plt.savefig("figures/top20_compostos.png", dpi = 300)
plt.show()

# Top 20 cinases predites més promíscues:
n_actives_per_kinase = preds_binary.sum(axis = 0)
df_promiscuous = pd.DataFrame({'target_chembl_id': kinase_names, 'n_actives_pred': n_actives_per_kinase})

df_promiscuous = df_promiscuous.merge(df_kinases, on = 'target_chembl_id')
df_promiscuous = df_promiscuous.sort_values('n_actives_pred', ascending = False)

top20_kinases = df_promiscuous.head(20)
plt.figure(figsize = (14, 6))
plt.barh(top20_kinases['pref_name'][::-1], top20_kinases['n_actives_pred'][::-1], color = 'xkcd:grey green', 
         edgecolor = 'white', height = 1.0)
plt.xlabel('Nombre de compostos predits com a actius')
plt.title(f'Top 20 cinases predites més promíscues amb FNN (threshold = {threshold})')
plt.tight_layout()
plt.savefig("figures/top20_cinases.png", dpi = 300)
plt.show()

# Heatmap:
top20_kinase_idx = [kinase_names.index(k) for k in df_promiscuous['target_chembl_id'].head(20).tolist()]
heatmap_data = preds[top20_idx][:, top20_kinase_idx]
kinase_short_names = df_promiscuous['pref_name'].head(20).tolist()

plt.figure(figsize = (14, 6))
sns.heatmap(heatmap_data, xticklabels = kinase_short_names, yticklabels = top20_ids, cmap = 'YlGn', vmin = 0,
    vmax = 1, linewidths = 0.3, annot = False)
plt.title('Probabilitats predites: Top 20 compostos vs Top 20 cinases')
plt.xlabel('Cinases')
plt.ylabel('ChEMBL ID compostos')
plt.xticks(rotation = 45, ha = 'right', fontsize = 8)
plt.yticks(fontsize = 8)
plt.tight_layout()
plt.savefig("figures/heatmap.png", dpi = 300)
plt.show()
