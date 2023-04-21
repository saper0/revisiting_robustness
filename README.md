# Revisiting Robustness in Graph Machine Learning

This codebase has been used to generate all results found in the ICLR 2023 paper *Revisiting Robustness in Graph Machine Learning*. 

The following instructions refer to the codebase found in the [src](src/) folder and have been used to generate all results pertaining random graph models (contextual stochastic block models / contextual Barabási-Albert models with community structure).

The codebase for the experiments on real graphs with a separate installation and usage description can be found in the folder [real_graphs](real_graphs/).

# Installation

The code requires the following packages and has been tested with the given versions:

```
python 3.9.7
pytorch 1.10.2
cudatoolkit 11.3.1
pyg 2.0.3
sacred 0.8.2
tqdm 4.62.3
scipy 1.7.3
torchtyping 0.1.4
seml 0.3.6
jupyterlab 3.2.9
numba 0.54.1
pytest 7.0.0 (optional: only for performing unit tests)
```

# Experiments

All experiments use [seml](https://github.com/TUM-DAML/seml). For an introduction into *seml*, we refer to the [official examples](https://github.com/TUM-DAML/seml/tree/master/examples) on the *seml* github repository. 

## Analysing (Over-) Robustness

The code in `exp_eval_robustness.py` trains a models with the provided hyperparameters and then, analysis its classic as well as semantic-aware robustness. The corresponding *seml* experiment configuration files can be found in [config/eval_robustness](config/eval_robustness/). 

Exemplary, the corresponding configuration file for the GCN+LP architecture is `csbm_gcn_lp.yaml`. It can be run by executing

```
seml [mongodb-collection-name] add config/eval_robustness/csbm_gcn_lp.yaml start
```

again, optionally adding the `--local` flag. 

Individually configured experiments can also be run without using the *seml* command-line interface and without requiring a MongoDB using the provided [exp.ipynb](exp.ipynb). 

## Hyperparameter Search

The experiment code to do a hyperparameter search for a given model is collected in `exp_train.py`. The corresponding *seml* experiment configuration files can be found in [config/training](config/training).

Exemplary, the corresponding configuration file for the GCN architecture is `csbm_gcn.yaml`. Training the GCNs can then be performed by executing:

```
seml [mongodb-collection-name] add config/training/csbm_gcn.yaml start
```

Optionally, the experiments can be run locally by adding the `--local` flag:

```
seml [mongodb-collection-name] add config/training/csbm_gcn.yaml start --local
```


# Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{
    gosch2023revisiting,
    title={Revisiting Robustness in Graph Machine Learning},
    author={Lukas Gosch and Daniel Sturm and Simon Geisler and Stephan G{\"u}nnemann},
    booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
    year={2023},
}
```

# Contact

For questions and feedback, please do not hesitate to contact:

Lukas Gosch, lukas (dot) gosch (at) tum (dot) de, Technical University of Munich

# Other Notes

This codebase contains code snippets from the following repositories:

- [Adversarial Attacks on Neural Networks for Graph Data](https://github.com/danielzuegner/nettack)
- [Robustness of Graph Neural Networks at Scale](https://github.com/sigeisler/robustness_of_gnns_at_scale)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)

We thank the authors for making their code public and the development team of *PyTorch Geometric* as well as *seml*.
