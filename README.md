# JT-VAE for the Dual-Objective Inverse Design of Metal Complexes

This repo contains the modified [JT-VAE](https://github.com/Bibyutatsu/FastJTNNpy3) code for the publication ["Deep Generative Model for the Dual-Objective Inverse Design of Metal Complexes."](https://doi.org/10.26434/chemrxiv-2024-mzs7b)

## Requirements

The [environment.yml](environment.yml) file is an export of a conda environment that can run this model.

**Important**:
The version of RDKit is very important. For newer versions of RDKit the model does not work!
The tree decomposition will give kekulization errors with newer versions of RDKit.

## Code for model training

- `fast_molvae/` contains codes for unconditional JT-VAE training. Please refer to `fast_molvae/README.md` for details.
- `fast_jtnn/` contains codes for model and data implementation.
- `fast_molopt/` contains codes for training a conditional JT-VAE and for performing conditional optimization with a trained model.
- `data/` contains various ligand training data.

#### FastJTNNpy3

The code is based on a fork of [FastJTNNpy3](https://github.com/Bibyutatsu/FastJTNNpy3).
