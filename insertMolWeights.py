#!/usr/bin/env python 

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

mols = pd.read_csv('SantaCruztBioTechStockListCarbohydratesFiltered_v4.csv')
# https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html#rdkit.Chem.rdMolDescriptors.CalcExactMolWt
mw = [rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(i)) for i in mols['SMILES']]
mols['Molecular Weight'] = mw
mols.rename(columns={'Unnamed: 0': 'Number'}, inplace=True)
mols.to_csv('SantaCruztBioTechStockListCarbohydratesFiltered_v5.csv',index=False)