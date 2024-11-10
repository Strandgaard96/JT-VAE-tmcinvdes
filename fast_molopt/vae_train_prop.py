import argparse
import logging
import os
import sys
import time
from pathlib import Path

from preprocess_prop import process_mol_trees

from fast_jtnn.datautils_prop import MolTreeDataset

sys.path.append("../")

import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from fast_jtnn.jtmpn import JTMPN
from fast_jtnn.jtnn_enc import JTNNEncoder
from fast_jtnn.mpn import MPN

source = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, str(source))

# Initialize logger
_logger: logging.Logger = logging.getLogger(__name__)

from fast_jtnn import JTpropVAE, MolTreeFolder_prop, Vocab, set_batch_nodeID


def get_git_revision_short_hash() -> str:
    """get the git hash of current commit if git repo."""
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def custom_collate_fn(batch):
    """Here we combine the tensors across the entries in a batch such that the
    batch dimensions are (10,(10,4)"""
    mol_tree_list = []
    property_list = []

    for i, item in enumerate(batch):
        mol_tree, property_tensor = item
        mol_tree_list.append(mol_tree)
        property_list.append(property_tensor)

    # Stack the tensor elements to create a batch property tensor of shape (batch_size, 4)
    property_stacked_tensor = torch.stack(property_list, dim=0)

    return mol_tree_list, property_stacked_tensor


def main_vae_train(
    args,
):
    #  Load the vocab
    with open(args.vocab_path) as f:
        vocab = f.read().splitlines()
    # vocab = [x.strip("\r\n ") for x in open(args.vocab)]
    vocab = Vocab(vocab)

    output_dir = Path(f"train_{time.strftime('%Y%m%d-%H%M%S')}")
    output_dir.mkdir(exist_ok=True)
    args.save_dir.mkdir(exist_ok=True)

    # Setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "printlog.txt"), mode="w"),
            logging.StreamHandler(),  # For debugging. Can be removed on remote
        ],
    )

    # Unpack beta
    beta = args.beta

    model = JTpropVAE(
        vocab,
        args.hidden_size,
        args.latent_size,
        args.depthT,
        args.depthG,
        args.train_mode,
    ).cuda()
    _logger.info(model)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    if args.load_previous_model:
        model.load_state_dict(torch.load(args.model_path))

    _logger.info(
        (
            "Model #Params: %dK"
            % (sum([x.nelement() for x in model.parameters()]) / 1000,)
        )
    )

    # Write commandline args to file
    with open(output_dir / "opts.txt", "w") as file:
        file.write(f"{vars(args)}\n")

    # Write the current git commit to the log
    _logger.info(f"Git commit tag: {get_git_revision_short_hash()}\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)

    total_step = args.load_epoch

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

    dataset = MolTreeDataset(
        train_path=args.dataset_path,
        prop_path=args.dataset_prop,
        vocab_path=args.vocab_path,
        developer_mode=args.developer_mode,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    for epoch in tqdm(list(range(args.epoch)), position=0, leave=True):
        for out in loader:
            # As of now, the featurization of each batch is performed here
            feature_tensors = process_trees(out[0], vocab, assm=True, optimize=False)
            batch = out + feature_tensors

            total_step += 1
            model.zero_grad()
            loss, log_metrics = model(batch, beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            # Log the current metrics
            if total_step % args.print_iter == 0:
                log_list = (
                    [f"Loss: {loss:.2f}", f" Beta: {beta:.5f}"]
                    + [f" {k}: {v:.3f}" for k, v in log_metrics.items()]
                    + [
                        f"PNorm: {model.param_norm:.3f}",
                        f"GNorm: {model.grad_norm:.3f}",
                    ]
                )
                log_string = ",".join(log_list)
                _logger.info(log_string)

            if total_step % args.save_iter == 0:
                torch.save(model.state_dict(), output_dir / f"model.iter-{total_step}")

            if total_step % args.anneal_iter == 0:
                scheduler.step()
                _logger.info(("learning rate: %.6f" % scheduler.get_lr()[0]))

            # Update the beta value
            if total_step % args.kl_anneal_iter == 0 and total_step >= args.warmup:
                beta = min(args.max_beta, beta + args.step_beta)

    torch.save(model.state_dict(), output_dir / f"model.epoch-{epoch}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, type=Path)
    parser.add_argument("--dataset_prop", required=True, type=Path)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--save_dir", required=True, type=Path)
    parser.add_argument("--load_previous_model", action="store_true")
    parser.add_argument("--developer_mode", action="store_true")
    parser.add_argument("--model_path", required=False, type=Path)
    parser.add_argument("--load_epoch", type=int, default=0)
    parser.add_argument(
        "--train_mode",
        nargs="*",
        default=[],
        choices=["denticity", "isomer"],
        help="Selects which extra property terms to include in the training, when using the argument each extra term should be separated by a space",
    )

    # These should not be touched
    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--batch_size", type=int, default=8)  # 2 when debugging)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip_norm", type=float, default=50.0)

    # These might need to be adjusted for a new problem
    parser.add_argument("--beta", type=float, default=0.006)
    parser.add_argument("--step_beta", type=float, default=0.002)
    parser.add_argument("--max_beta", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--anneal_rate", type=float, default=0.9)
    parser.add_argument("--anneal_iter", type=int, default=1000)
    parser.add_argument("--kl_anneal_iter", type=int, default=3000)

    parser.add_argument("--epoch", type=int, default=150)
    parser.add_argument("--print_iter", type=int, default=1)
    parser.add_argument("--save_iter", type=int, default=5000)

    args = parser.parse_args()
    _logger.info(args)

    main_vae_train(
        args=args,
    )
