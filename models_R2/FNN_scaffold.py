import numpy as np
import torch
import torch.nn as nn
from scaffold_split import scaffold_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment
import random

# Reproduïbilitat:
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Càrrega de dades
X = np.load("data/morgan_fingerprints.npy").astype(np.float32)
Y = np.load("data/Y_matrix.npy").astype(np.float32)

# Partició:
train, val, test = scaffold_split("data/chembl_smiles.csv", val_size = 0.10, test_size = 0.20)

X_train, X_val, X_test = X[train], X[val], X[test]
Y_train, Y_val, Y_test = Y[train], Y[val], Y[test]

# Conversió a tensors:
device = "cuda" if torch.cuda.is_available() else "cpu"

X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).float()
X_val = torch.from_numpy(X_val).float()
Y_val = torch.from_numpy(Y_val).float()
X_test = torch.from_numpy(X_test).float()
Y_test = torch.from_numpy(Y_test).float()

train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size = 256, shuffle = True)
val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size = 256, shuffle = False)
test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size = 256, shuffle = False)

# Feed Forward Neural Network:
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

FNN_scaffold_model = FeedForward_NN(2048, Y_train.shape[1]).to(device)

# Funció mask:
def masked_loss(prediction, target):                 
    mask = ~torch.isnan(target)                      

    loss = nn.BCEWithLogitsLoss(reduction = 'none')  
    loss_value = loss(prediction, torch.nan_to_num(target)) 

    masked_loss = loss_value * mask                   

    if mask.sum() == 0:                               
        return torch.zeros((), device = prediction.device, requires_grad = True)

    return masked_loss.sum() / mask.sum()

# Funció per al càlcul de les mètriques d'avaluació:
def evaluate(loader):
    FNN_scaffold_model.eval()

    total_preds = []
    y_true_labels = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            preds = torch.sigmoid(FNN_scaffold_model(x_batch)).cpu() 

            total_preds.append(preds)
            y_true_labels.append(y_batch)

    total_preds = torch.cat(total_preds).numpy()
    y_true_labels = torch.cat(y_true_labels).numpy()

    roc_auc_scores = []
    pr_auc_scores = []
    bedroc_scores = []
    enrich_factor = []

    for target in range(y_true_labels.shape[1]):
        true_target_label = y_true_labels[:, target]
        target_probs = total_preds[:, target]

        mask = ~np.isnan(true_target_label)
        true_target_label = true_target_label[mask]
        target_probs = target_probs[mask]

        if len(np.unique(true_target_label)) < 2: 
            continue

        roc_auc_scores.append(roc_auc_score(true_target_label, target_probs))
        pr_auc_scores.append(average_precision_score(true_target_label, target_probs))

        order = np.argsort(-target_probs)
        scores_array = np.column_stack([target_probs[order], true_target_label[order]])

        bedroc_scores.append(CalcBEDROC(scores_array, col = 1, alpha = 20.0))
        enrich_factor.append(CalcEnrichment(scores_array, col = 1, fractions = [0.01])[0])

    return {
        "AUC": np.nanmean(roc_auc_scores),
        "PR-AUC": np.nanmean(pr_auc_scores),
        "BEDROC": np.nanmean(bedroc_scores),
        "EF1%": np.nanmean(enrich_factor),
        "Targets": len(roc_auc_scores)
    }

# Optimitzador Adam amb learning rate de 1e-4:
optimizer = torch.optim.Adam(FNN_scaffold_model.parameters(), lr = 1e-4, weight_decay = 1e-4)

# Entrenament del model:
epochs = 100
best_auc = 0
patience = 10
no_improve = 0

for epoch in range(epochs):

    FNN_scaffold_model.train()
    total_loss = 0

    for X_batch, Y_batch in train_loader:

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        valid_data = (~torch.isnan(Y_batch)).sum(dim = 1) > 0 
        X_batch = X_batch[valid_data]
        Y_batch = Y_batch[valid_data]

        if X_batch.shape[0] == 0:
            continue

        optimizer.zero_grad()

        predictions = FNN_scaffold_model(X_batch)
        loss_values = masked_loss(predictions, Y_batch)

        loss_values.backward()
        optimizer.step()

        total_loss += loss_values.item()

 # Avaluació:
    results_val = evaluate(val_loader)               

    print(
        f"Epoch {epoch} | Loss: {total_loss:.4f} | "
        f"ROC-AUC: {results_val['AUC']:.4f} | "
        f"PR-AUC: {results_val['PR-AUC']:.4f} | "
        f"BEDROC: {results_val['BEDROC']:.4f} | "
        f"EF1%: {results_val['EF1%']:.4f}")

# Early Stopping:
    if results_val["AUC"] > best_auc:
        best_auc = results_val["AUC"]
        no_improve = 0
        torch.save(FNN_scaffold_model.state_dict(), "models/best_FNN_scaffold.pt")
    else:
        no_improve += 1

    if no_improve >= patience:
        print("Early stopping")
        break

FNN_scaffold_model.load_state_dict(torch.load("models/best_FNN_scaffold.pt"))
results_test = evaluate(test_loader) 

# Resultats:
print("Resultats")
for k, v in results_test.items():
    if k != "Targets":
        print(f"{k}: {v:.4f}")
print("Targets evaluated:", results_test["Targets"])