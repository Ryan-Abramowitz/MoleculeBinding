import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier # horrible because inbalanced classes
from sklearn.linear_model import LogisticRegression # sucks
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import NuSVC
from sklearn.svm import SVC
import shlex
import subprocess
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, TorsionFingerprints
from rdkit.Chem import Descriptors
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import Draw
from multiprocessing import Pool
import math
import timeit
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
import math
from typing import Callable, Dict, List, Set, Tuple

import numpy as np


import random
random.seed(0)

testsmiles = 'O[C@H](C)C(=O)O'
testname = 'D-(-)-Lactic acid'







df = pd.read_excel('SupplimentaryTables.xlsx',sheet_name='Sheet1')



CLEAN_EVERYTHING = True

if CLEAN_EVERYTHING:
    def clean_molecule(m): # https://github.com/rdkit/rdkit/issues/46 (because I'm using this on broken substructures)
        Chem.SanitizeMol(m,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
else:
    def clean_molecule(m): # https://github.com/rdkit/rdkit/issues/46 (because I'm using this on broken substructures)
        Chem.SanitizeMol(m,sanitizeOps=Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)






bits = int(2048)

smiles = df['SMILES']
names = df['Names']
molecs = [Chem.MolFromSmiles(i) for i in smiles]
[clean_molecule(i) for i in molecs]
[i.SetProp("_Name",j) for i,j in zip(molecs,names)]
molecs = [i for i in molecs if not i == None]
fingerprint = np.array([AllChem.GetMorganFingerprintAsBitVect(i,2,useChirality=True,nBits=bits) for i in molecs])


label = df['Label']
tot = len(label)
pos = sum(label)
neg = tot - sum(label)




# print(df)
print(f"FULL Dataset: {df.shape}")
print("Fingerprint:",fingerprint.shape,"Sparcity:",np.mean(fingerprint))


# successes = 3 close, 5
# 1, 5, 7
# gone = smiles.index('')
print(f"{testname}:\t{testsmiles}")

individual = lambda x: SGDClassifier(random_state=x,loss='modified_huber', penalty='elasticnet', max_iter=1000, class_weight={0: 0.005, 1: 1}, warm_start=True, early_stopping=False, n_jobs=-1)#SGDClassifier(random_state=x,loss='modified_huber', penalty='elasticnet', validation_fraction=sys.float_info.min,n_jobs=-1)
getmodel = lambda: VotingClassifier([(str(i),individual(i)) for i in range(10)], voting= 'soft', n_jobs=-1)

short_finger = list(fingerprint)
short_label = list(label)
model = getmodel()
print("model ready!")
# model.fit(short_finger,short_label)
model.fit(fingerprint,label)

# models = [(str(i),SGDClassifier(random_state=i,loss='log', validation_fraction=sys.float_info.min, penalty='elasticnet')) for i in range(10)]
# model = VotingClassifier(models, voting= 'soft', n_jobs=-1)
# # model = SGDClassifier(random_state=0,loss='log', validation_fraction=sys.float_info.min, penalty='elasticnet',n_jobs=-1)
# model.fit(short_finger,[i for i in label if i != gone])
print("fit model!")

creativity = 0
def model_on_list(x):
    result = []
    for s in x:
        fingerprint = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s),2,useChirality=True,nBits=bits)])
        result.append(model.predict_proba(fingerprint)[0][1]*(1-creativity)+0.5*creativity)
    return result
pred = (model_on_list([testsmiles])[0] -0.5*creativity)/(1-creativity)
print(f"prediction: {round(pred,2)}")

pcut = 0.5
rollouts = 100
if pred < pcut:
    print("WARNING: FULL MOLECULE PREDICTED NOT TO BIND\n\tSCORE\t= ",round(pred,2))
    
print("running monty carlo search...")



def submolecules(molecule):
    submolecules = set([molecule])
    molecule = Chem.MolFromSmiles(molecule, sanitize=False)
    clean_molecule(molecule)
    
    atoms = [i for i in molecule.GetAtoms() if i.GetAtomicNum() != 1]
    atoms.sort(key=lambda x: x.GetIdx(),reverse=True)
    numatoms = len(atoms)
    print(f'Generating {2**numatoms - 2} submolecules using {numatoms} atoms')
    for subinds in range(1,(2**numatoms)-1):
        deletions = [((subinds//2**i)%(2**(i+1))) == 0 for i in range(numatoms)]
        # print(deletions)
        new_mol = Chem.EditableMol(molecule)
        for ind,delete in enumerate(deletions):
            if delete:
                for bond in atoms[ind].GetNeighbors():
                    new_mol.RemoveBond(atoms[ind].GetIdx(),bond.GetIdx())
        for ind,delete in enumerate(deletions):
            if delete:
                new_mol.RemoveAtom(atoms[ind].GetIdx())
        new_mol = Chem.MolToSmiles(new_mol.GetMol())
        if not new_mol in submolecules:
            print('.',end='')
            submolecules.add(new_mol)
    print(f'\nReturning {len(submolecules)} unique submolecules')
    return submolecules



main_mols = []
legend = []



pred = model_on_list([testsmiles])[0]
print(f"{testname}:\t{testsmiles}\t--\tscore = {round(pred,2)}")

starter = Chem.MolFromSmiles(testsmiles, sanitize=False)
clean_molecule(starter)
substructs = 2**len([i for i in starter.GetAtoms() if i.GetAtomicNum() != 1]) - 2
if substructs > 10**5:
    print(f"{substructs} substructures is to many :'(\nIt's to many to seach through\nThis should work on molecules with less atoms!")
else:
    pcut = 0.5

    if pred < pcut:
        print("WARNING: FULL MOLECULE PREDICTED NOT TO BIND\n\tSCORE = ",round(pred,2))

    # print("running exhaustive search...")
    subs = submolecules(testsmiles)
    # print(f"found {len(subs)} submolecules...:{subs}")
    passing_subs = [(i,j,) for i,j in zip(model_on_list(subs),subs) if i > pcut]
    print(f"Found {len(passing_subs)} posative submolecules!")

    template = Chem.MolFromSmiles(testsmiles)
    AllChem.Compute2DCoords(template)
    # for i,n in enumerate(passing_subs):
    #     print(f"Submolecule {i}:\tScore = {round(n[0],2)}\t{n[1]}")
    mMols = [Chem.MolFromSmiles(testsmiles, sanitize=False)] + [Chem.MolFromSmiles(m[1], sanitize=False) for m in passing_subs]
    clean_molecule(mMols[0])
    # if pred < pcut:
    #     print("WARNING: FULL MOLECULE PREDICTED NOT TO BIND\n\tSCORE = ",round(pred,2))
        
    print('Done!')

    fragments = [mMols[0].GetSubstructMatches(m,maxMatches=100000000,useChirality=True) for m in mMols[1:]]
    for at in mMols[0].GetAtoms():
        overlaps = [max([at.GetIdx() in f for f in i]+[0]) for i in fragments]
        at.SetProp('atomNote',f"{round(sum(overlaps)/(len(fragments)+sys.float_info.min),2)}")
        
    main_mols.append(mMols[0])
    legend.append(f"{testsmiles}\n{testname}\n{len(fragments)} successfull submolecules")

    print(f"CLEAN_EVERYTHING-{CLEAN_EVERYTHING}")

    x = Draw.MolsToGridImage(main_mols,legends=legend, maxMols=100000000, molsPerRow=5, subImgSize=(250,250), returnPNG=False, useSVG=True)
    with open('junk.svg','w+') as f:
        f.write(x.data)
    Draw.MolsToGridImage(main_mols,legends=legend, maxMols=100000000, molsPerRow=5, subImgSize=(250,250), useSVG=True)