import sys

sys.path.append("../")
import argparse
import sys

import rdkit
import torch

from fast_jtnn import *


def load_model(vocab, model_path, hidden_size=450, latent_size=56, depthT=20, depthG=3):
    vocab = [x.strip("\r\n ") for x in open(vocab)]
    vocab = Vocab(vocab)

    # model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model = JTpropVAE(
        vocab,
        int(hidden_size),
        int(latent_size),
        int(opts.depthT),
        int(opts.depthG),
        train_mode=["denticity"],
    )
    dict_buffer = torch.load(model_path)
    model.load_state_dict(dict_buffer)
    model = model.cuda()

    torch.manual_seed(1)
    return model


def main_sample(
    vocab,
    output_file,
    model_path,
    nsample,
    hidden_size=450,
    latent_size=56,
    depthT=20,
    depthG=3,
):
    vocab = [x.strip("\r\n ") for x in open(vocab)]
    vocab = Vocab(vocab)

    # model = JTNNVAE(vocab, hidden_size, latent_size, depthT, depthG)
    model = JTpropVAE(
        vocab,
        int(hidden_size),
        int(latent_size),
        int(depthT),
        int(depthG),
        train_mode=["denticity"],
    )
    dict_buffer = torch.load(model_path)
    model.load_state_dict(dict_buffer)
    model = model.cuda()

    torch.manual_seed(2)
    with open(output_file, "w") as out_file:
        for i in range(nsample):
            if i % 10 == 0:
                print(i)
            out_file.write(str(model.sample_prior(prob_decode=False)) + "\n")


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()
    parser.add_argument("--nsample", type=int, required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--hidden_size", type=int, default=450)
    parser.add_argument("--latent_size", type=int, default=56)
    parser.add_argument("--depthT", type=int, default=20)
    parser.add_argument("--depthG", type=int, default=3)

    args = parser.parse_args()
    main_sample(
        args.vocab,
        args.output_file,
        args.model,
        args.nsample,
        args.hidden_size,
        args.latent_size,
        args.depthT,
        args.depthG,
    )
