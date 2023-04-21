# Overrobustness of GNNs on real-world datasets

This codebase has been used to generate all results based on the real-world datasets in the ICLR 2023 paper *Revisiting Robustness in Graph Machine Learning*.

## Installation

The repository requires the python package [rgnn_at_scale](https://github.com/sigeisler/robustness_of_gnns_at_scale). Please follow the [installation instructions](https://github.com/sigeisler/robustness_of_gnns_at_scale#installation) to install the package as well as other required packages. The custom coda kernels of [rgnn_at_scale](https://github.com/sigeisler/robustness_of_gnns_at_scale) are not required.

## Training

The experiment `experiment_train_inductive.py` performs inductive training for the in the config specified models. The corresponding config files can be found in [config/train](config/train/)

To train the models run 
```bash
python script_execute_experiment.py --config-file 'config/evaluate/cora_ml_and_citeseer.yaml'
```

The trained models are stored in the `cache` folder and the output is logged into `./output`.

## Evaluation

The experiment `experiment_evaluate_overrobustness.py` evaluates the overrobustness for the models saved in `cache`. The corresponding config files can be found in [config/evaluate](config/evaluate/)

To run the evaluation run 
```bash python script_execute_experiment.py --config-file 'config/evaluate/cora_ml_and_citeseer.yaml'```

The output is logged into `./output`.

# Other Notes

This codebase contains code snippets from the following repositories:

- [Robustness of Graph Neural Networks at Scale](https://github.com/sigeisler/robustness_of_gnns_at_scale)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)

We thank the authors for making their code public and the development team of *PyTorch Geometric*.
