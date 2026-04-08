TFM: Predicting Off-Target Effects of Kinase Inhibitors Using Multi-Label Classification

Descripció del projecte i objectius

Aquest projecte té com a objectiu entrenar i comparar dos models de ML per predir els efectes *off-target* de diversos compostos sobre un grup seleccionat de cinases, enzims transferases que catalitzen la fosforilació de substrats específics i són crucials en processos com el metabolisme, el creixement i la divisió cel·lular. La conservació del lloc d’unió de l’ATP fa que molts inhibidors de cinases puguin afectar altres membres de la família, generant efectes adversos com la cardiotoxicitat.

Aquest treball busca aportar un mètode robust per predir interaccions *off-target* en cinases, destacant la rellevància de la classificació *multi-Label* per identificar correlacions entre interaccions i efectes adversos. Els resultats proporcionaran eines útils per al disseny de fàrmacs més segurs i eficients en fases inicials del descobriment farmacològic.

Les dades es recullen de la base de dades [ChEMBL](https://www.ebi.ac.uk/chembl/), desenvolupant-se models de ML amb Scikit-Learn i models de DL amb PyTorch, utilitzant una classificació *multi-Label* per predir les interaccions múltiples de cada compost. Els models es compararan mitjançant mètriques específiques per determinar quin enfocament ofereix millor rendiment predictiu.

Aquest repositori inclou el pipeline de descàrrega i preprocessament de les dades dut a terme fins ara.
---

Estructura del repositori


data/
   db.tsv/                 # Base de dades inicial descarregada de ChEMBL
   db_filtered.csv/        # Dataset filtrat amb mínim 50 compostos i 700 activitats
   db_final.csv/           # Dataset amb els compostos, les activitats i l'assay_ID
   conf_assays_types.csv/  # Llistat de les cinases amb els confidence_score, l'assay_ID i l'assay_type
   db_complete.csv/        # Dataset amb tota la informació: molecule_chembl_id, pchembl_value, standard_type,       
                             standard_value,target_chembl_id,assay_chembl_id,confidence_score,assay_type
src/                  
   chembl_activities.py    # Codi de la descàrrega dels compostos, les activitats i l'assay_ID associats a cada target.
   chembl_conf_score.py    # Codi de la descàrrega del confidence_score, l'assay_type i l'assay_ID associats a cada target. 
   merge_complete.py       # Unió dels dos llistats obtinguts mitjançant el paràmetre assay_ID.

figures/              
   Boxplot_distribució_activitats.png

README.md             