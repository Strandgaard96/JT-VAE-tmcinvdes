# Conditional JT-VAE training

## Important notes

The property JT-VAE from the original JT-VAE repo was not compatible with the rest of the code. I adapted it to work with the new data processing.

## Training of conditional JT-VAE

Is done in the same way as the unconditional JT-VAE (see file fast_molvae/README.md)

## Optimization in latent space

To do directional optimization in latent space run optimize.py.
This script need a input csv file with encoded SMILES and their corresponding DFT labeled properties.
It also needs a trained model and vocabulary.

An example of running the optimization is:

```
python -u fast_molopt/optimize.py input_dir_path data/example_prompt_ligands.csv --vocab_path vocab.txt --cutoff 0.2 --model_path $model
```
