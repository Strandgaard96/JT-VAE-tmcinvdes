import itertools
import os
import pickle as pickle
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fast_jtnn.jtmpn import JTMPN
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.mpn import MPN
from fast_jtnn.vocab import Vocab
from fast_molopt.preprocess_prop import (
    load_smiles_and_props_from_files,
    process_mol_trees,
)

# class MolTreeDataset(Dataset):
#     def __init__(
#         self,
#         train_path: str,
#         prop_path: str,
#         vocab_path: str,
#         developer_mode=False,
#     ):
#         """Constructor for the MolTrees.
#
#         Arguments:
#             root (str): The directory path in which to store raw and processed data.
#             developer_mode (bool): If set to True will only consider 1000 first data points.
#         """
#
#         # directory to getdataset_dirraw data from
#         self.train_path = train_path
#         self.root = train_path.parent.name
#         self.prop_path = prop_path
#         self._raw_dir = self.root + "/raw/"
#         self.assm = True
#         self.optimize = False
#
#         vocab = [x.strip("\r\n ") for x in open(vocab_path)]
#         self.vocab = Vocab(vocab)
#
#         # if developer mode is set to True will only consider first 1000 examples
#         self.developer_mode = developer_mode
#
#         # if root path does not exist create folder
#         if not os.path.isdir(self.root):
#             os.makedirs(self.root, exist_ok=True)
#
#         # start super class
#         super().__init__()
#
#         self.preprocess()
#
#         with open(self.train_path.parent / "processed.pickle", "rb") as fh:
#             self.trees, self.props = pickle.load(fh)
#         if self.developer_mode:
#             self.trees = self.trees[0:100]
#             self.props = self.props[0:100]
#
#     def __len__(self):
#         """Getter for the number of processed pytorch graphs."""
#         return len(self.trees)
#
#     def __getitem__(self, idx):
#         return tensorize_prop(
#             self.trees[idx],
#             self.props[idx],
#             self.vocab,
#             assm=self.assm,
#             optimize=self.optimize,
#         )
#
#     def preprocess(self):
#         all_data, prop_data = process_mol_trees(
#             train_path=self.train_path,
#             prop_path=self.prop_path,
#             njobs=6,
#             developer_mode=self.developer_mode,
#         )
#         all_data = np.array(all_data)
#
#         prop_data = torch.FloatTensor(prop_data)
#
#         with open(os.path.join(self.train_path.parent / "processed.pickle"), "wb") as f:
#             pickle.dump((all_data, prop_data), f)


class MolTreeDataset(Dataset):
    def __init__(
        self,
        smiles_path,
        properties_path,
        vocab_path,
        batch_size,
        cache_dir="cache/batches",
        developer_mode=False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size

        vocab = [x.strip("\r\n ") for x in open(vocab_path)]
        self.vocab = Vocab(vocab)

        # Load SMILES and properties data
        self.smiles_list, self.properties_array = load_smiles_and_props_from_files(
            smiles_path, properties_path, developer_mode
        )

        # Check if cache exists; if not, create it
        if not self.cache_dir.exists() or not list(self.cache_dir.glob("batch_*.pt")):
            print("Cache not found. Creating batch cache...")
            self.cache_batches()

        # Load list of batch files
        self.batch_files = sorted(self.cache_dir.glob("batch_*.pt"))

    def cache_batches(self):
        """Processes data in batches and saves each batch to a separate file in
        cache_dir."""
        num_samples = len(self.smiles_list)
        num_batches = (
            num_samples + self.batch_size - 1
        ) // self.batch_size  # Calculate total number of batches

        for batch_idx in tqdm(range(num_batches), desc="Caching batches", unit="batch"):
            # Determine the range of indices for this batch
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_samples)

            # Process each item in the batch

            mol_trees = process_mol_trees(self.smiles_list[start_idx:end_idx], njobs=6)
            property_tensors = torch.tensor(
                self.properties_array[start_idx:end_idx], dtype=torch.float32
            )

            # Process mol_tree to generate required tensors
            jtenc_holders, mpn_holders, jtmpn_data = process_trees(
                mol_trees, self.vocab
            )
            jtmpn_holders, batch_idx_tensor = jtmpn_data

            # # Append processed data to batch lists
            # mol_trees.append(mol_tree)
            # property_tensors.append(property_tensor)
            # jtenc_holders.append(jtenc_holder)
            # mpn_holders.append(mpn_holder)
            # jtmpn_holders.append(jtmpn_holder)
            # batch_idxs.append(batch_idx_tensor)

            # Save the entire batch to a single file
            batch_data = {
                "mol_trees": mol_trees,
                "property_tensors": property_tensors,
                "jtenc_holders": jtenc_holders,
                "mpn_holders": mpn_holders,
                "jtmpn_holders": jtmpn_holders,
                "batch_idxs": batch_idx_tensor,
            }
            torch.save(batch_data, self.cache_dir / f"batch_{batch_idx}.pt")

        print(f"Cached {num_batches} batches to {self.cache_dir}")

    def __len__(self):
        return len(self.batch_files)

    def __getitem__(self, idx):
        # Load the entire batch from a single file
        batch_data = torch.load(self.batch_files[idx])

        # Return batch data directly as a tuple, ready for the model
        return (
            batch_data["mol_trees"],
            batch_data["property_tensors"],
            batch_data["jtenc_holders"],
            batch_data["mpn_holders"],
            (batch_data["jtmpn_holders"], batch_data["batch_idxs"]),
        )


class MolTreeFolder_prop(object):
    def __init__(
        self,
        data_folder,
        vocab,
        batch_size,
        num_workers=4,
        shuffle=False,
        assm=True,
        replicate=None,
        optimize=False,
    ):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        self.optimize = optimize

        if replicate is not None:  # expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn, "rb") as f:
                data = pickle.load(f)

            if self.shuffle:
                random.shuffle(data)  # shuffle data before batch

            data, prop_data = data
            batches = [
                data[i : i + self.batch_size]
                for i in range(0, len(data), self.batch_size)
            ]
            batches_prop = [
                prop_data[i : i + self.batch_size]
                for i in range(0, len(prop_data), self.batch_size)
            ]

            dataset = MolTreeDataset_prop(
                batches, batches_prop, self.vocab, self.assm, self.optimize
            )
            dataloader = DataLoader(
                dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x[0]
            )  # , num_workers=self.num_workers)

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader


class MolTreeDataset_prop(Dataset):
    def __init__(self, data, prop_data, vocab, assm=True, optimize=False):
        self.data = data
        self.prop_data = prop_data
        self.vocab = vocab
        self.assm = assm
        self.optimize = optimize

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return tensorize_prop(
            self.data[idx],
            self.prop_data[idx],
            self.vocab,
            assm=self.assm,
            optimize=self.optimize,
        )


def tensorize_prop(tree_batch, prop_batch, vocab, assm=True, optimize=False):
    # set_batch_nodeID(tree_batch, vocab)
    prop_batch_tensor = torch.FloatTensor(prop_batch)
    return tree_batch, prop_batch_tensor


def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


def process_trees(tree_batch, vocab, assm=True, optimize=False):
    "This function performs extra featurization of the mol trees that is needed during training"
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            # Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1:
                continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return (
        jtenc_holder,
        mpn_holder,
        (jtmpn_holder, batch_idx),
    )
