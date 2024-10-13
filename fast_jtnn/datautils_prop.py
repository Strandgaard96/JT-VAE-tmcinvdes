import itertools
import os
import pickle as pickle
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fast_jtnn.jtmpn import JTMPN
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.mpn import MPN
from fast_jtnn.vocab import Vocab
from fast_molopt.preprocess_prop import process_mol_trees


class MolTreeDataset(Dataset):
    def __init__(
        self,
        train_path: str,
        prop_path: str,
        vocab_path: str,
        developer_mode=False,
    ):
        """Constructor for the MolTrees.

        Arguments:
            root (str): The directory path in which to store raw and processed data.
            developer_mode (bool): If set to True will only consider 1000 first data points.
        """

        # directory to getdataset_dirraw data from
        self.train_path = train_path
        self.root = train_path.parent.name
        self.prop_path = prop_path
        self._raw_dir = self.root + "/raw/"
        self.assm = True
        self.optimize = False

        vocab = [x.strip("\r\n ") for x in open(vocab_path)]
        self.vocab = Vocab(vocab)

        # if developer mode is set to True will only consider first 1000 examples
        self.developer_mode = developer_mode

        # if root path does not exist create folder
        if not os.path.isdir(self.root):
            os.makedirs(self.root, exist_ok=True)

        # start super class
        super().__init__()

        self.preprocess()

        with open(self.train_path.parent / "processed.pickle", "rb") as fh:
            self.trees, self.props = pickle.load(fh)
        if self.developer_mode:
            self.trees = self.trees[0:100]
            self.props = self.props[0:100]

    def __len__(self):
        """Getter for the number of processed pytorch graphs."""
        return len(self.trees)

    def __getitem__(self, idx):
        return tensorize_prop(
            self.trees[idx],
            self.props[idx],
            self.vocab,
            assm=self.assm,
            optimize=self.optimize,
        )

    def preprocess(self):
        all_data, prop_data = process_mol_trees(
            train_path=self.train_path,
            prop_path=self.prop_path,
            njobs=6,
            developer_mode=True,
        )
        with open(os.path.join(self.train_path.parent / "processed.pickle"), "wb") as f:
            pickle.dump((all_data, prop_data), f)


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
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = tree_batch.smiles
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    # TL debug, with https://stackoverflow.com/a/70323486 {
    # print(prop_batch)
    # print(type(prop_batch))
    # prop_batch = np.vstack(prop_batch).astype(np.float)
    # print(prop_batch)
    # print(type(prop_batch))
    # torch.from_numpy(a)

    prop_batch_tensor = torch.FloatTensor(prop_batch)

    if assm is False:
        return tree_batch, prop_batch_tensor, jtenc_holder, mpn_holder

    cands = []
    # for i, mol_tree in enumerate(tree_batch):
    for node in tree_batch.nodes:
        # Leaf node's attachment is determined by neighboring node's attachment
        if node.is_leaf or len(node.cands) == 1:
            continue
        cands.extend([(cand, tree_batch.nodes, node) for cand in node.cands])

    # WARNING: THIS WAS ADDED WHEN DOING LOCAL OPTIMIZATION.
    # the holder throws an error when doing local optimization.
    if optimize:
        jtmpn_holder = None
    else:
        jtmpn_holder = JTMPN.tensorize(cands, mess_dict)

    return (
        tree_batch,
        prop_batch_tensor,
        jtenc_holder,
        mpn_holder,
        jtmpn_holder,
    )


def set_batch_nodeID(mol_tree, vocab):
    tot = 0
    # for mol_tree in mol_batch:
    for node in mol_tree.nodes:
        node.idx = tot
        node.wid = vocab.get_index(node.smiles)
        tot += 1
