import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
import rdkit
import rdkit.Chem as Chem
import torch
import torch.nn as nn
from rdkit.Chem import Descriptors
from torch.autograd import Variable

from fast_jtnn import *

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))
from fast_molopt.preprocess_prop import main_preprocess

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = argparse.ArgumentParser()
parser.add_argument("--test_path", required=True)
parser.add_argument("--vocab_path", required=True)
parser.add_argument("--model_path", required=True)
parser.add_argument("--output_path", default="optimize-processed")
parser.add_argument("--prop_path", default=True)
parser.add_argument("--nsplits", type=int, default=2)

parser.add_argument("--hidden_size", type=int, default=450)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--latent_size", type=int, default=56)
parser.add_argument("--depthT", type=int, default=20)
parser.add_argument("--depthG", type=int, default=3)
parser.add_argument("--cutoff", type=float, default=0.5)

parser.add_argument("--lr", type=float, default=1.5)
parser.add_argument("--clip_norm", type=float, default=50.0)
parser.add_argument("--beta", type=float, default=0.0)
parser.add_argument("--step_beta", type=float, default=0.002)
parser.add_argument("--max_beta", type=float, default=1.0)
parser.add_argument("--warmup", type=int, default=500)

parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--anneal_rate", type=float, default=0.9)
parser.add_argument("--anneal_iter", type=int, default=1000)
parser.add_argument("--kl_anneal_iter", type=int, default=3000)
parser.add_argument("--print_iter", type=int, default=50)
parser.add_argument("--save_iter", type=int, default=1000)


opts = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
sim_cutoff = float(opts.cutoff)

model = JTpropVAE(
    vocab, int(hidden_size), int(latent_size), int(opts.depthT), int(opts.depthG)
).cuda()
print(model)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()


# data = []
# with open(opts.test_path) as f:
#     for line in f:
#         s = line.strip("\r\n ").split()[0]
#         data.append(s)

res = []


# Preprocess data
main_preprocess(opts.test_path, opts.prop_path, opts.output_path, opts.nsplits)

loader = MolTreeFolder_prop(
    opts.output_path, vocab, batch_size=1, shuffle=False
)  # , num_workers=4)


results = defaultdict(list)

output_dir = Path(f"opt_{time.strftime('%Y%m%d-%H%M%S')}")
output_dir.mkdir(exist_ok=True)

with open(output_dir / "opts.json", "w") as file:
    file.write(json.dumps(vars(opts)))

for batch in loader:
    # Extract smiles
    smiles = batch[0][0].smiles

    p = batch[1].clone().detach().cpu().numpy().squeeze()
    results["props"].append(p)
    results["smiles"].append(smiles)

    # z_tree_vecs, z_mol_vecs = model.encode_from_smiles([smiles], prop_batch = [0.12697000000000003,0.3016386189305794])
    # new_smiles1 = model.decode(z_tree_vecs, z_mol_vecs, prob_decode=False)
    # z_tree_vecs, z_mol_vecs = model.encode_from_smiles([smiles], prop_batch = [0.127,0.301])
    # tree_vec, x_tree_mess, mol_vec = self.encode(x_jtenc_holder, x_mpn_holder)
    # new_smiles = model.decode(z_tree_vecs, z_mol_vecs, prob_decode=False)
    mol = Chem.MolFromSmiles(smiles)
    new_smiles = model.sample_prior(prob_decode=False)
    # new_smiles = model.sample_prior_conditional(prompt_condition=[0.126,0.3],prob_decode=False)

    # new_smiles, sim = model.conditional_optimization(
    #     batch, sim_cutoff=sim_cutoff, lr=opts.lr, num_iter=200, type="first"
    # )
    new_smiles, sim = model.optimize(
        batch, sim_cutoff=sim_cutoff, lr=opts.lr, num_iter=200, type="first"
    )
    print(new_smiles)
    new_mol = Chem.MolFromSmiles(new_smiles)
    results["new_smiles"].append(new_smiles)

df = pd.DataFrame(data=results)
df.to_csv(output_dir / "optimize_results.csv", index=False)
