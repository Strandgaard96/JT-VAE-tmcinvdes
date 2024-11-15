# Accelerated Training of Junction Tree VAE

The training sets for the model can be found at : [Training sets](https://github.com/uiocompcat/tmcinvdes/tree/main/datasets/01_tmQMg-L-training_sets)

To train a model run the following scripts sequentially in the conda environment given in environment.yml
Replace train.txt with the path to the training set you want to use.

```
# # run tree composition and create vocab
python -u fast_jtnn/mol_tree.py -i train.txt -v vocab.txt

# Train unconditional model
python -u fast_molvae/vae_train.py --dataset_path train.txt --vocab vocab.txt --save_dir vae_model/
```

Default Options:

`--beta 0` means to set KL regularization weight (beta) initially to be zero.

`--warmup 2` means that beta will not increase within the first 2 epochs. Warmup is recommended as using large KL regularization (large beta) in the beginning of training is harmful for model performance.

`--step_beta 0.002 --kl_anneal_iter 5` means beta will increase by 0.002 every 5th epoch. You should observe that the KL term will decrease as beta increases.

`--max_beta 1.0 ` sets the maximum value of beta to be 1.0.

Please note that this is not necessarily the best annealing strategy. You are welcomed to adjust these parameters. Additionally, the parameters are dataset size dependant.
A larger dataset means that the number of iterations will be higher before the whole dataset is run trough once (1 epoch).
