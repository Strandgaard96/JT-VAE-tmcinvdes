# JT-VAE: Deep Generative Model for the Dual-Objective Inverse Design of Metal Complexes

This repo contains the modified JT-VAE code for the publication ["Deep Generative Model for the Dual-Objective Inverse Design of Metal Complexes."](https://doi.org/10.26434/chemrxiv-2024-mzs7b)

This repo is a fork of the repo : ["Python 3 Version of Fast Junction Tree Variational Autoencoder for Molecular Graph Generation (ICML 2018)"](https://github.com/Bibyutatsu/FastJTNNpy3)

## Requirements

The version of RDKit is very important. For newer versions of RDKit the model does not work!
The tree decomposition will give kekulization errors with newer versions of RDKit.
The [environment.yml](environment.yml) file is an export of a working conda environment that can run this model.

## Code for model training

- `fast_molvae/` contains codes for unconditional JT-VAE training. Please refer to `fast_molvae/README.md` for details.
- `fast_jtnn/` contains codes for model implementation.
- `fast_molopt/` contains codes for training a conditional JT-VAE.
