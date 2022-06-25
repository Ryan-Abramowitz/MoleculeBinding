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

gone = 7








df = pd.read_excel('SupplimentaryTables.xlsx',sheet_name='Sheet1')







MIN_ATOMS = 1
C_PUCT = 10


class MCTSNode:
    """A :class:`MCTSNode` represents a node in a Monte Carlo Tree Search."""

    def __init__(self, smiles: str, atoms: List[int], W: float = 0, N: int = 0, P: float = 0) -> None:
        """
        :param smiles: The SMILES for the substructure at this node.
        :param atoms: A list of atom indices represented by this node.
        :param W: The W value of this node.
        :param N: The N value of this node.
        :param P: The P value of this node.
        """
        self.smiles = smiles
        self.atoms = set(atoms)
        self.children = []
        self.W = W
        self.N = N
        self.P = P

    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0

    def U(self, n: int) -> float:
        return C_PUCT * self.P * math.sqrt(n) / (1 + self.N)


def find_clusters(mol: Chem.Mol) -> Tuple[List[Tuple[int, ...]], List[List[int]]]:
    """
    Finds clusters within the molecule.

    :param mol: An RDKit molecule.
    :return: A tuple containing a list of atom tuples representing the clusters
             and a list of lists of atoms in each cluster.
    """
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:  # special case
        return [(0,)], [[0]]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append((a1, a2)) # any 2 atoms that are bonded

    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)] # any rings
    clusters.extend(ssr)

    atom_cls = [[] for _ in range(n_atoms)]
    for i in range(len(clusters)):
        for atom in clusters[i]:
            atom_cls[atom].append(i)

    return clusters, atom_cls


def __extract_subgraph(mol: Chem.Mol, selected_atoms: Set[int]) -> Tuple[Chem.Mol, List[int]]:
    """
    Extracts a subgraph from an RDKit molecule given a set of atom indices.

    :param mol: An RDKit molecule from which to extract a subgraph.
    :param selected_atoms: The atoms which form the subgraph to be extracted.
    :return: A tuple containing an RDKit molecule representing the subgraph
             and a list of root atom indices from the selected indices.
    """
    selected_atoms = set(selected_atoms)
    roots = []
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        bad_neis = [y for y in atom.GetNeighbors() if y.GetIdx() not in selected_atoms]
        if len(bad_neis) > 0:
            roots.append(idx)

    new_mol = Chem.RWMol(mol)

    for atom_idx in roots:
        atom = new_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        aroma_bonds = [bond for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        aroma_bonds = [bond for bond in aroma_bonds if
                       bond.GetBeginAtom().GetIdx() in selected_atoms and bond.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:
            atom.SetIsAromatic(False)

    remove_atoms = [atom.GetIdx() for atom in new_mol.GetAtoms() if atom.GetIdx() not in selected_atoms]
    remove_atoms = sorted(remove_atoms, reverse=True)
    for atom in remove_atoms:
        new_mol.RemoveAtom(atom)

    return new_mol.GetMol(), roots


def extract_subgraph(smiles: str, selected_atoms: Set[int]) -> Tuple[str, List[int]]:
    """
    Extracts a subgraph from a SMILES given a set of atom indices.

    :param smiles: A SMILES from which to extract a subgraph.
    :param selected_atoms: The atoms which form the subgraph to be extracted.
    :return: A tuple containing a SMILES representing the subgraph
             and a list of root atom indices from the selected indices.
    """
    # try with kekulization
    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
#     print("1",end = '')
    subgraph, roots = __extract_subgraph(mol, selected_atoms)
#     print("2",end = '')
    try:
#         print("3",end = '')
        subgraph = Chem.MolToSmiles(subgraph, kekuleSmiles=True)
#         print("4",end = '')
        subgraph = Chem.MolFromSmiles(subgraph)
    except Exception:
        subgraph = None

#     print("5",end = '')
    mol = Chem.MolFromSmiles(smiles)  # de-kekulize
#     print("6",end = '')
    if subgraph is not None and mol.HasSubstructMatch(subgraph):
        return Chem.MolToSmiles(subgraph), roots

    # If fails, try without kekulization
#     print("7",end = '')
    subgraph, roots = __extract_subgraph(mol, selected_atoms)
#     print("8",end = '')
    subgraph = Chem.MolToSmiles(subgraph)
    subgraph = Chem.MolFromSmiles(subgraph)

    if subgraph is not None:
        return Chem.MolToSmiles(subgraph), roots
    else:
        return None, None


def mcts_rollout(node: MCTSNode,
                 state_map: Dict[str, MCTSNode],
                 orig_smiles: str,
                 clusters: List[Set[int]],
                 atom_cls: List[Set[int]],
                 nei_cls: List[Set[int]],
                 scoring_function: Callable[[str], List[float]],
                 prop_delta: float) -> float:
    """
    A Monte Carlo Tree Search rollout from a given :class:`MCTSNode`.

    :param node: The :class:`MCTSNode` from which to begin the rollout.
    :param state_map: A mapping from SMILES to :class:`MCTSNode`.
    :param orig_smiles: The original SMILES of the molecule.
    :param clusters: Clusters of atoms.
    :param atom_cls: Atom indices in the clusters.
    :param nei_cls: Neighboring clusters.
    :param scoring_function: A function for scoring subgraph SMILES using a Chemprop model.
    :return: The score of this MCTS rollout.
    """
    cur_atoms = node.atoms
#     print(f"Rolling out on {len(cur_atoms)} atoms!")
    if len(cur_atoms) <= MIN_ATOMS:
#         print(f"Molecule to small... {node.P} | {node.smiles}")
        return [node.P, node.P, node.smiles]

    # Expand if this node has never been visited
    if len(node.children) == 0:
        cur_cls = set([i for i, x in enumerate(clusters) if x <= cur_atoms])
        for i in cur_cls:
#             print(".", end = '')
            leaf_atoms = [a for a in clusters[i] if len(atom_cls[a] & cur_cls) == 1]
            if len(nei_cls[i] & cur_cls) == 1 or len(clusters[i]) == 2 and len(leaf_atoms) == 1:
                new_atoms = cur_atoms - set(leaf_atoms)
#                 print("-", end = '')
                new_smiles, _ = extract_subgraph(orig_smiles, new_atoms)
#                 print("-", end = '')
                if new_smiles in state_map:
                    new_node = state_map[new_smiles]  # merge identical states
                else:
                    new_node = MCTSNode(new_smiles, new_atoms)
                if new_smiles:
                    node.children.append(new_node)

        state_map[node.smiles] = node
        if len(node.children) == 0:
#             print(f"no children:    {node.P} | {node.smiles}")
            return [node.P, node.P, node.smiles]  # cannot find leaves

        scores = scoring_function([x.smiles for x in node.children])
        for child, score in zip(node.children, scores):
            child.P = score
#             print(f"child score:    {child.P} | {child.smiles}")
#     print("\n",end = '')
    sum_count = sum(c.N for c in node.children)
    selected_node = max(node.children, key=lambda x: x.Q() + x.U(sum_count))
    v = mcts_rollout(selected_node, state_map, orig_smiles, clusters, atom_cls, nei_cls, scoring_function, prop_delta)
    selected_node.W += v[0]
#     selected_node.W += v[1]
    selected_node.N += 1
    if v[1] < prop_delta:
        v[1:] = [node.P, node.smiles]
    return v


def mcts(smiles: str,
         scoring_function: Callable[[str], List[float]],
         n_rollout: int,
         prop_delta: float) -> List[MCTSNode]:
    """
    Runs the Monte Carlo Tree Search algorithm.

    :param smiles: The SMILES of the molecule to perform the search on.
    :param scoring_function: A function for scoring subgraph SMILES using a Chemprop model.
    :param n_rollout: The number of MCTS rollouts to perform.
    :param prop_delta: The minimum required property value for a satisfactory rationale.
    :return: A list of rationales each represented by a :class:`MCTSNode`.
    """
    
    mol = Chem.MolFromSmiles(smiles)
    if mol.GetNumAtoms() > 50:
        n_rollout = 1

    clusters, atom_cls = find_clusters(mol)
    nei_cls = [0] * len(clusters)
    # List of atoms in each cluster
    for i, cls in enumerate(clusters):
        # The set of cluster that this cluster intersects except itself
        nei_cls[i] = [nei for atom in cls for nei in atom_cls[atom]]
        nei_cls[i] = set(nei_cls[i]) - {i}
        # This cluster
        clusters[i] = set(list(cls))
    # List of clusters for each atom
    for a in range(len(atom_cls)):
        atom_cls[a] = set(atom_cls[a])

    # W = N = P = 0
    root = MCTSNode(smiles, set(range(mol.GetNumAtoms())))
    state_map = {smiles: root}
    min_smiles = []
    for _ in range(n_rollout):
        v = mcts_rollout(root, state_map, smiles, clusters, atom_cls, nei_cls, scoring_function, prop_delta)
        if v[1] > prop_delta:
            min_smiles.append(tuple(v[1:]))
#         print(f"rollout finale: {v}")

#     rationales = [node for _, node in state_map.items() if len(node.atoms) <= max_atoms and node.P >= prop_delta]

    return min_smiles

bits = int(2048)

smiles = df['SMILES']
names = df['Names']
molecs = [Chem.MolFromSmiles(i) for i in smiles]
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
print(f"{names[gone]}:\t{smiles[gone]}")

individual = lambda x: SGDClassifier(random_state=x,loss='log', penalty='elasticnet', validation_fraction=sys.float_info.min,n_jobs=-1)
getmodel = lambda: VotingClassifier([(str(i),individual(i)) for i in range(10)], voting= 'soft', n_jobs=-1)

short_finger = list(fingerprint[:gone]) + list(fingerprint[gone+1:])
short_label = list(label[:gone]) + list(label[gone+1:])
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
pred = (model_on_list([smiles[gone]])[0] -0.5*creativity)/(1-creativity)
print(f"prediction: {round(pred,2)}")

pcut = 0.5
rollouts = 100
if pred < pcut:
    print("WARNING: FULL MOLECULE PREDICTED NOT TO BIND\n\tSCORE\t= ",round(pred,2))
    
print("running monty carlo search...")
min_molecule = mcts(
    smiles[gone],
    model_on_list,
    rollouts, # number of rollouts
    pcut, # probability cutoff
)

template = Chem.MolFromSmiles(smiles[gone])
AllChem.Compute2DCoords(template)
for i,n in enumerate(min_molecule):
    print(f"Submolecule {i}:\tScore = {round(n[0],2)}\t{n[1]}")
mMols = [Chem.MolFromSmiles(smiles[gone])] + [Chem.MolFromSmiles(m[1]) for m in min_molecule]

if pred < pcut:
    print("WARNING: FULL MOLECULE PREDICTED NOT TO BIND\n\tSCORE = ",round(pred,2))
    
print('Done!')

fragments = [mMols[0].GetSubstructMatch(m) for m in mMols[1:]]
for at in mMols[0].GetAtoms():
    overlaps = [at.GetIdx() in f for f in fragments]
    at.SetProp('atomNote',f"{round(sum(overlaps)/len(overlaps),2)}")
    
main_mol = mMols[0]
# print(f"{smiles[gone]}")
# print(f"{names[gone]}")
# print(f"{len(fragments)} rollouts")
# Draw.MolsToGridImage(mMols,legends=[names[gone]]+[f'Submolecule {i+1}/{len(fragments)}' for i in range(len(mMols)-1)], subImgSize=(250,250), useSVG=False)
Draw.MolsToGridImage(mMols[:1],legends=[names[gone]+f"\n{len(fragments)} successfull rollouts"], subImgSize=(250,250), useSVG=False)
# main_mol

