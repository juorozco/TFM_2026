# Off-Target Activity Prediction Across Human Kinases
### A Comparative Study of SVM and Feedforward Neural Networks Using ChEMBL Bioactivity Data

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11.0-red)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.8.0-orange)
![RDKit](https://img.shields.io/badge/RDKit-2026.3.1-green)
![NumPy](https://img.shields.io/badge/NumPy-2.4.4-blue)
![Pandas](https://img.shields.io/badge/Pandas-3.0.2-purple)
![ChEMBL](https://img.shields.io/badge/ChEMBL_client-0.10.9-lightgrey)

## Overview

This repository contains the code and data for the Master's Thesis (TFM) on predicting off-target interactions of chemical compounds across 318 human kinases using bioactivity data from the ChEMBL database.

Two model types are compared under two data splitting strategies:
- **Support Vector Machine (SVM)** — trained independently per kinase (Scikit-Learn)
- **Feedforward Neural Network (FNN)** — multi-task, trained simultaneously on all targets (PyTorch)
- **Random split** and **Scaffold split** (Bemis–Murcko)

All models are evaluated using ROC-AUC, PR-AUC, BEDROC (α = 20) and Enrichment Factor at 1% (EF1%).

---

## Repository Structure
---

## Repository Structure

```
TFM/
├── data/
│   ├── partials/                   # Intermediate data files
│   ├── chembl_smiles.csv           # Final SMILES dataset
│   ├── db_complete.csv             # Processed activity dataset
│   ├── morgan_fingerprints.npy     # Morgan fingerprints radius 2 (Zenodo)
│   ├── morgan_fingerprints_R3.npy  # Morgan fingerprints radius 3 (Zenodo)
│   └── Y_matrix.npy                # Binary activity matrix (Zenodo)
├── figures/
│   ├── Boxplot_acts_filter.png
│   └── distribucio_pchembl.png
├── src/
│   ├── best_models/                # Trained model files (.pt, .pkl)
│   ├── data_download/
│   │   ├── chembl_activities.py    # Download bioactivity data from ChEMBL
│   │   ├── chembl_conf_score.py    # Download assay confidence scores
│   │   ├── chembl_smiles.py        # Download SMILES for compounds
│   │   ├── morgan_fingerP.py       # Generate Morgan fingerprints (radius 2)
│   │   └── morgan_fingerP_R3.py    # Generate Morgan fingerprints (radius 3)
│   ├── data_preparation/
│   │   ├── merge_and_quality.py    # Merge datasets and apply quality filters
│   │   └── multilabel_matrix.py    # Build binary activity matrix
│   ├── models_R2/                  # Models with radius = 2
│   │   ├── random_split/
│   │   │   ├── SVM_random.py
│   │   │   └── FNN_random.py
│   │   └── scaffold_split/
│   │       ├── SVM_scaffold.py
│   │       └── FNN_scaffold.py
│   ├── models_R3/                  # Models with radius = 3
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

## Author

**Judith Orozco**
Master's Thesis — Universitat Oberta de Catalunya (UOC), 2026