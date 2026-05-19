# Off-Target Activity Prediction Across Human Kinases
### A Comparative Study of Support Vector Machine, Random Forest and Feedforward Neural Networks Using ChEMBL Bioactivity Data

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-red)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.8.0-orange)
![RDKit](https://img.shields.io/badge/RDKit-2026.3.1-green)
![NumPy](https://img.shields.io/badge/NumPy-2.4.4-blue)
![Pandas](https://img.shields.io/badge/Pandas-3.0.2-purple)
![ChEMBL](https://img.shields.io/badge/ChEMBL_client-0.10.9-lightgrey)

## Descripció

Aquest repositori conté el codi i les dades del treball de final de màster (TFM) sobre la predicció d'interaccions off-target de. compostos químics sobre 318 cinases humanes, utilitzant dades de bioactivitat extretes de la base de dades ChEMBL.

Es comparen tres tipus de models sota dues estratègies de partició de dades:
- **Support Vector Machine (SVM) i Random Forest (RF)** — entrenats de forma independent per a cada cinasa (Scikit-Learn)
- **Feedforward Neural Network (FNN)** — multi-task, entrenada simultàniament sobre tots els targets (PyTorch)
- **Random split** and **Scaffold split** (Bemis–Murcko)

Tots els models s'avaluen mitjançant les mètriques ROC-AUC, PR-AUC, BEDROC (α = 20) i Enrichment Factor a l'1% (EF1%).

---

## Estructura del repositori

```
TFM/
├── data/
│   ├── partials/                   # Fitxers de dades intermedis
│   ├── chembl_smiles.csv           # Dataset d'SMILES finals
│   ├── db_complete.csv             # Dataset d'activitat processat
│   ├── morgan_fingerprints.npy     # Morgan fingerprints radi 2
│   ├── morgan_fingerprints_R3.npy  # Morgan fingerprints radi 3
│   └── Y_matrix.npy                # Matriu d'activitat binària
├── figures/
│   ├── Boxplot_acts_filter.png
│   └── distribucio_pchembl.png
├── src/
│   ├── best_models/                # Millors models entrenats
│   ├── data_download/
│   │   ├── chembl_activities.py    # Descàrrega de dades de bioactivitat de ChEMBL
│   │   ├── chembl_conf_score.py    # Descàrrega dels graus de confiança
│   │   ├── chembl_smiles.py        # Descàrrega dels SMILES dels compostos
│   │   ├── morgan_fingerP.py       # Generació de Morgan fingerprints (radi 2)
│   │   └── morgan_fingerP_R3.py    # Generació de Morgan fingerprints (radi 3)
│   ├── data_preparation/
│   │   ├── merge_and_quality.py    # Fusió de les dades i implementació dels filtres de qualitat
│   │   └── multilabel_matrix.py    # Construcció de la matriu d'activitat
│   ├── models_R2/                  # Models amb radi = 2
│   │   ├── random_split/
│   │   │   ├── SVM_random.py
│   │   │   └── FNN_random.py
│   │   └── scaffold_split/
│   │       ├── SVM_scaffold.py
│   │       └── FNN_scaffold.py
│   ├── models_R3/                  # Models amb radi = 3
│   │   ├── random_split/
│   │   │   ├── SVM_R3.py
│   │   │   └── FNN_R3.py
│   │   └── scaffold_split/
│   │       ├── SVM_scaffold_R3.py
│   │       └── FNN_scaffold_R3.py
│   └── scaffold_split.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Autora

**Judith Orozco**
— Treball de final de màster -- Universitat Oberta de Catalunya (UOC) i UB, 2026