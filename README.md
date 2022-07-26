# MoleculeBinding

This repository contains python scripts for the classification model and accompanying data as described from the paper:
Exploring the substrate specificity of a sugar transporter with biosensors and cheminformatics

This is a binary classification to predict which molecules are recognized by the plant membrane transporter AtSWEET1 through biosensor SweetTrac1:
https://www.pnas.org/doi/abs/10.1073/pnas.2119183119
The SGD classifier from sklearn was used for the binary classifier and morgan fingerprints for the molecules were generated using RDKit suite:

1.	G. Landrum (2006) RDKit: open-source cheminformatics.
2.	F. Pedregosa et al., Scikit-learn: machine learning in python. Journal of Machine Learning Research 12, 2825–2830 (2011).

## The contents of this repository:
  - Classification Model.py was used to train the model and predictmolecule binding. 
  - Model Scorer.py was used to train and generate the confusion matrix (3-fold cross validation)
  - SupplementaryTables.xlsx includes all chemicals for the primary, secondary, and tertiary screen, their SMILES, and their classification as a hit      (1) or non-hit (0). 
  - Model Prediction trained on Primary and Secondary screens.xlsx is the predicted scores of all of the compounds after the model was trained on the       primary and secondary screen. 
  - Model Prediction trained on all screens.xlsx is the predicted scores of all the compounds after the model was trained on the primary, secondary,       and tertiary screen. 
  - SantaCruzBioTech Catalog Score.xlsx is the predicted scores of the 400 pre-screened compounds from SantaCruzBioTech's carbohydrate's catalog that
    had their SMILES available and was under 400 g/mol. Highlighted in green were the 14 selected compounds that were not duplicates from the previous    screens, were affordable and considered a hit ( > 0.5)
