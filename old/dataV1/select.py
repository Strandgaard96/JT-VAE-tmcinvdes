import sys

import numpy as np
import rdkit
import sascorer
from rdkit.Chem import Descriptors, MolFromSmiles, MolToSmiles, rdmolops

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

smiles = []
for line in sys.stdin:
    smiles.append(line.strip())

targets = []
for i in range(len(smiles)):
    logp = Descriptors.MolLogP(MolFromSmiles(smiles[i]))
    sa = sascorer.calculateScore(MolFromSmiles(smiles[i]))
    targets.append(logp - sa)

smiles = list(zip(smiles, targets))
smiles = sorted(smiles, key=lambda x: x[1])
for x, y in smiles:
    print(x, y)
