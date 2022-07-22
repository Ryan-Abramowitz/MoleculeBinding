#!/usr/bin/env python 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from rdkit import Chem
from rdkit.Chem import AllChem

df = pd.read_excel('SupplimentaryTables.xlsx', sheet_name='Sheet1')

bits = int(2048)
def clean_molecule(m): # https://github.com/rdkit/rdkit/issues/46 (because I'm using this on broken substructures)
    Chem.SanitizeMol(m,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)

smiles = df['SMILES']
names = df['Name']
molecs = [Chem.MolFromSmiles(i, sanitize=False) for i in smiles]
[clean_molecule(s) for s in molecs]
[i.SetProp("_Name",j) for i,j in zip(molecs,names)]
molecs = [i for i in molecs if not i == None]
fingerprint = np.array([AllChem.GetMorganFingerprintAsBitVect(i,2,useChirality=True,nBits=bits) for i in molecs])

label = df['Label']
tot = len(label)
pos = sum(label)
neg = tot - sum(label)

features = np.all(fingerprint == 0,axis=0) + np.all(fingerprint == 1,axis=0) == 0
print('Features:',features.sum())

print()
print(f"FULL Dataset: {df.shape}")
print("Fingerprint:",fingerprint.shape,"Sparcity:",np.mean(fingerprint))


df = pd.read_excel('SupplimentaryTables.xlsx', sheet_name='Sheet3')
nonsmiles = []
for i,j in enumerate(df['SMILES']):
    try:
        x = Chem.MolFromSmiles(j, sanitize=False)
        clean_molecule(x)
        AllChem.GetMorganFingerprintAsBitVect(x,2,useChirality=True,nBits=bits)
    except:
        nonsmiles.append(i)
df.drop(nonsmiles, axis=0, inplace=True)

baselines = [np.zeros((np.sum(features),)), np.ones((np.sum(features),))]
inp = [np.eye(np.sum(features),np.sum(features)),1-np.eye(np.sum(features),np.sum(features))]

nums = []
bases = []
for _ in range(1000):
    model = SGDClassifier(loss='modified_huber', penalty='elasticnet', max_iter=1000, class_weight={0: 0.005, 1: 1}, warm_start=True, early_stopping=False, n_jobs=-1)
    model.fit(fingerprint[:,features],label)
    def model_on_list(fingerprints):
        return model.predict_proba(fingerprints)[:,1]
    predictions = model_on_list(inp)
    nums += [predictions]
    predictions = model_on_list(baselines)
    bases += [predictions]
    print('.',end='')
nums = np.array(nums)
bases = np.array(bases)
importants = nums.mean(axis=1)
baselines = bases.mean(axis=1)

plt.plot(importants[0] - baselines[0],'.r')
plt.plot(baselines[1] - importants[1],'.b')
plt.title('Test Predictions')
plt.ylabel('Feature Importants')
plt.xlabel('Feature Number')
plt.show()
