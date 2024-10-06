import argparse
import sys

sys.path.append("../")
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from tqdm import tqdm

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))
from fast_jtnn import *
from fast_jtnn.datautils_prop import *


def create_mol_tree(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


def load_smiles_and_props_from_files(train_path, prop_path):
    """Loads smiles and properties.

    Args:
        train_path (Path): Path to SMILES txt file
        prop_path (Path): Path to csv file with properties and property headers

    Returns:
        smiles,prop_data
    """
    with open(train_path) as f:
        smiles = [line.strip("\r\n ").split()[0] for line in f]
    print("Input File read")

    prop_data = pd.read_csv(prop_path, sep=",")
    print(f"Input property file read with the properties: {list(prop_data.columns)}")
    print(
        "NB! This column ordering gives the ordering of properties in the property tensor"
    )

    # Verify that the number of properties match the number of data points
    if len(smiles) != len(prop_data):
        raise Exception(
            "TEMRINATED, number of lines in property file does not match number of smiles"
        )

    return smiles, prop_data


def process_mol_trees(train_path, prop_path, num_splits, pickle_output_path, njobs):
    # Crate ouput dir
    out_path = os.path.join(pickle_output_path, "./")
    if os.path.isdir(out_path) is False:
        os.makedirs(out_path)

    smiles, prop_data = load_smiles_and_props_from_files(
        train_path=train_path,
        prop_path=prop_path,
    )

    print("Converting SMILES to MolTrees.....")
    pool = Pool(njobs)
    all_data = pool.map(create_mol_tree, smiles)

    all_data_split = np.array_split(all_data, num_splits)
    prop_data_split = np.array_split(prop_data.to_numpy(), num_splits)
    print("MolTree processsing Complete")

    print("Storing mol trees in pickles")
    for split_id in tqdm(list(range(num_splits)), position=0, leave=True):
        with open(
            os.path.join(pickle_output_path, "tensors-%d.pkl" % split_id), "wb"
        ) as f:
            pickle.dump((all_data_split[split_id], prop_data_split[split_id]), f)

    return True


def main_preprocess(train_path, prop_path, output_path, num_splits=10, njobs=1):
    process_mol_trees(train_path, prop_path, num_splits, output_path, njobs)
    return True


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", dest="train_path")
    parser.add_argument("-p", "--prop_path", dest="prop_path")
    parser.add_argument("-n", "--split", dest="nsplits", default=10, type=int)
    parser.add_argument("-j", "--jobs", dest="njobs", default=8, type=int)
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="The folder to store the processed pickles",
    )

    opts = parser.parse_args()

    process_mol_trees(
        opts.train_path, opts.prop_path, opts.nsplits, opts.output_path, opts.njobs
    )
