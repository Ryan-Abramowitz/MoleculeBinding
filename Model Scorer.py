import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

# --- Clean and Convert Test data -----

df = pd.read_excel('Model Score Dataset.xlsx', sheet_name='Sheet1')
bits = int(2048)


def clean_molecule(m):  # https://github.com/rdkit/rdkit/issues/46 (because I'm using this on broken substructures)
    Chem.SanitizeMol(m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)


def trainer(X_train, y_train, X_test, y_test, features):
    clf = SGDClassifier(loss='modified_huber', penalty='elasticnet', max_iter=1000, class_weight={0: 0.005, 1: 1},
                        warm_start=True, early_stopping=False, n_jobs=-1)

    nums = np.zeros((1, np.shape(X_test)[0]))

    for _ in range(1000):
        clf.fit(X_train[:, features], y_train)
        predictions = clf.predict_proba(X_test)[:, 1].reshape((1, np.shape(X_test)[0]))
        nums = np.concatenate((nums, predictions), axis=0)

    nums = nums[1:, :]
    y_pred = np.array(nums.mean(axis=0))

    return y_pred

# --- Cross validate prediction for obtaining average confusion matrix ----
def confusion_matrix_scorer(y_pred):

    y_pred_bin =  (y_pred >= thresh).astype(int)  # Marks predictions (0 - 1) larger than a threshold as true
    cm = confusion_matrix(y_test, y_pred_bin, labels=[1, 0])
    # [tp fn fp tn] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    score_result = np.array([cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]])

    return score_result


# --- get fingerprints ---
def get_fingerprint_train(molecs, names):
    [clean_molecule(s) for s in molecs]
    [i.SetProp("_Name", j) for i, j in zip(molecs, names)]
    molecs = [i for i in molecs if not i == None]

    fingerprint = np.array([AllChem.GetMorganFingerprintAsBitVect(i, 2, useChirality=True, nBits=bits) for i in molecs])
    features = np.all(fingerprint == 0, axis=0) + np.all(fingerprint == 1, axis=0) == 0

    return fingerprint, features


def get_fingerprint_test(molecs, names):
    [clean_molecule(s) for s in molecs]
    [i.SetProp("_Name", j) for i, j in zip(molecs, names)]
    molecs = [i for i in molecs if not i == None]

    fingerprint = np.array(
        [AllChem.GetMorganFingerprintAsBitVect(i, 2, useChirality=True, nBits=bits) for i in molecs])[:, features]
    return fingerprint


# --- Loop over threshold values ---
global thresh
tp, fn, fp, tn , threshold = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

label = df['Label']
smiles = df['SMILES']
names = df['Name']
molecs = [Chem.MolFromSmiles(i, sanitize=False) for i in smiles]


# --- loop over thresholds and k-splits of the data, repeatedly training and fitting the model

X,y = np.array(molecs), label

kf = KFold(n_splits=3)
kf.get_n_splits(X)

n = 50
tpr = np.zeros((n, 1))
fpr = np.zeros((n, 1))
tnr = np.zeros((n, 1))

for train_index, test_index in kf.split(X):

    # -- cross fold, splits ---
    smiles_train, smiles_test = smiles[train_index], smiles[test_index]
    names_train, names_test = names[train_index], names[test_index]
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Get fingerprints for these
    fgp_train, features = get_fingerprint_train(X_train, names_train)
    fgp_test = get_fingerprint_test(X_test, names_test)

    y_pred = trainer(fgp_train, y_train, fgp_test, y_test, features)

    score_temp = np.array([[0, 0, 0, 0]])
    for thresh in np.linspace(0, 1, num=n):
        score = np.array([confusion_matrix_scorer(y_pred)])
        score_temp = np.concatenate((score_temp, score), axis=0)

    score_temp = score_temp[1:, :]

    tp, fn, fp, tn = score_temp[:, 0].reshape((n,1)), score_temp[:, 1].reshape((n,1)), score_temp[:, 2].reshape((n,1)), score_temp[:, 3].reshape((n,1))

    tpr_temp = np.divide(tp, np.add(tp, fn))
    fpr_temp = np.divide(fp, np.add(fp, tn))
    tnr_temp = np.divide(tn, np.add(tn, fp))

    # record tpr and fpr for each loop:
    tpr = np.concatenate((tpr, tpr_temp), axis=1)
    fpr = np.concatenate((fpr, fpr_temp), axis=1)
    tnr = np.concatenate((tnr, tnr_temp), axis=1)
    thresh = np.linspace(0,1,n)

tpr, fpr, tnr = tpr[:, 1:], fpr[:, 1:], tnr[:, 1:]

# # --- Print All Values to a excel sheet ---
with pd.ExcelWriter('score rates.xlsx') as writer:
    for i in range(tpr.shape[1]):
        d_conf = {'Threshold': thresh, 'True positive rate': tpr[:,i], 'False Positive rate': fpr[:,i],'True negative rate': tnr[:,i]}
        df_conf = pd.DataFrame(data= d_conf)
        df_conf.to_excel(writer, sheet_name = 'Sheet_name_'+ str(i))

