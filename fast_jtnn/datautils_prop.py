from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from fast_jtnn.jtmpn import JTMPN
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.mpn import MPN
from fast_jtnn.vocab import Vocab
from fast_molopt.preprocess_prop import get_mol_trees, load_smiles_and_props_from_files


class MolTreeDataset(Dataset):
    "Class that processes data and supplies it to the JT-VAE during training"

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

        if properties_path:
            print("Detected property file. Loading SMILES and properties")
            # Load SMILES and properties data
            self.smiles_list, self.properties_array = load_smiles_and_props_from_files(
                smiles_path, properties_path, developer_mode
            )
            self.props = True
        else:
            self.props = False
            print("Loading SMILES")
            with open(smiles_path) as f:
                smiles = [line.strip("\r\n ").split()[0] for line in f]
            self.smiles_list = smiles

        # Check if cache exists; if not, create it
        if not self.cache_dir.exists() or not list(self.cache_dir.glob("batch_*.pt")):
            print("Cache not found. Creating batch cache...")
            self.cache_batches()

        # Chcek if cache is complete
        number_of_cached_files = len(list(self.cache_dir.glob("batch_*.pt")))
        expected_number_of_files = len(self.smiles_list) // batch_size

        if not number_of_cached_files != expected_number_of_files:
            raise Exception(
                "Corrupt cache. The number of batch files do not match the current dataset length and batch size. Please delete this directory and re-create the cached data."
            )

        # Load list of batch files
        self.batch_files = sorted(self.cache_dir.glob("batch_*.pt"))

    def cache_batches(self):
        """Processes data in batches and saves each batch to a separate file in
        cache_dir."""
        num_samples = len(self.smiles_list)
        num_batches = (
            num_samples + self.batch_size - 1
        ) // self.batch_size  # Calculate total number of batches
        print("Caching: ", num_samples, num_batches)
        print(num_batches, self.batch_size)

        for batch_idx in tqdm(range(num_batches), desc="Caching batches", unit="batch"):
            # Determine the range of indices for this batch
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_samples)

            mol_trees = get_mol_trees(self.smiles_list[start_idx:end_idx], njobs=6)
            set_batch_nodeID(mol_trees, self.vocab)

            if self.props:
                property_tensors = torch.tensor(
                    self.properties_array[start_idx:end_idx], dtype=torch.float32
                )
            else:
                property_tensors = None

            # Process mol_tree to generate required tensors
            jtenc_holders, mpn_holders, jtmpn_data = get_tensors(mol_trees)
            jtmpn_holders, batch_idx_tensor = jtmpn_data

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
        if self.props:
            return (
                batch_data["mol_trees"],
                batch_data["property_tensors"],
                batch_data["jtenc_holders"],
                batch_data["mpn_holders"],
                (batch_data["jtmpn_holders"], batch_data["batch_idxs"]),
            )
        else:
            return (
                batch_data["mol_trees"],
                batch_data["jtenc_holders"],
                batch_data["mpn_holders"],
                (batch_data["jtmpn_holders"], batch_data["batch_idxs"]),
            )


def set_batch_nodeID(mol_batch, vocab):
    "Utility function that sets ids that are used within the JT-VAE."
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1


def get_tensors(tree_batch, assm=True, optimize=False):
    "This function performs extra featurization of the mol trees that is needed during training"
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
