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
    mol_trees, property_tensors, jtenc_holders, mpn_holders, jtmpn_data = (
        [],
        [],
        [],
        [],
        [],
    )

    for mol_tree, property_tensor, jtenc, mpn, jtmpn in batch:
        mol_trees.append(mol_tree)
        property_tensors.append(property_tensor)
        jtenc_holders.append(jtenc)
        mpn_holders.append(mpn)
        jtmpn_data.append(jtmpn)

    # Stack tensors as needed
    jtenc_batch = torch.stack(jtenc_holders)
    mpn_batch = torch.stack(mpn_holders)
    jtmpn_holder, batch_idx = zip(*jtmpn_data)
    property_tensor_batch = torch.stack(property_tensors)

    # Return with property_tensor right after mol_tree
    return (
        mol_trees,
        property_tensor_batch,
        jtenc_batch,
        mpn_batch,
        (torch.stack(jtmpn_holder), torch.stack(batch_idx)),
    )


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

    # Initialize the dataset
    dataset = MolTreeDataset(
        smiles_path=args.dataset_path,
        properties_path=args.dataset_prop,
        vocab_path=args.vocab_path,  # Assume `vocab` is defined
        batch_size=args.batch_size,
        cache_dir="cache/batches",
    )

    # DataLoader with batch_size=1, since each file is a full batch
    loader = DataLoader(
        dataset,
        batch_size=1,  # Each item is already a batch
        shuffle=True,  # Shuffle to randomize batch order during training
        collate_fn=lambda x: x[0],  # Unwrap the single-item list returned by DataLoader
    )

    for epoch in tqdm(
        range(args.epoch), position=0, leave=True, desc="Training epochs"
    ):
        for batch in loader:
            # As of now, the featurization of each batch is performed here
            # feature_tensors = process_trees(out[0], vocab, assm=True, optimize=False)

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
                        f"PNorm: {model.param_norm():.3f}",
                        f"GNorm: {model.grad_norm():.3f}",
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
