"""
This experiment implements only training a model on a synthetic graph for
hyperparameter optimization.
"""
import logging
from typing import Any, Dict, Union, Optional

import numpy as np
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.run import Run
import torch

from src.data import split
from src.graph_models import create_graph_model
from src.models import create_model, LP
from src.train import train_inductive, train_transductive

try:
    import seml
    from seml.database import get_mongodb_config
except ModuleNotFoundError: 
    seml = None


ex = Experiment()
ex.captured_out_filter = apply_backspaces_and_linefeeds
if seml is not None:
    seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


def get_exp_name(data_params: Dict[str, Any],
                 model_params: Dict[str, Any],
                 train_params: Dict[str, Any],
                 seed) -> str:
    K = data_params["K"]
    exp_name = "trn_" + data_params["graph_model"] + f"_K{K:.1f}_"
    exp_name += "inductive" + str(train_params["inductive"]) + "_" 
    exp_name += model_params["label"] + f"_seed{seed}"
    return exp_name
    

@ex.config
def config():
    overwrite = None
    if seml is not None:
        db_collection = None
        if db_collection is not None:
            ex.observers.append(seml.create_mongodb_observer(db_collection, 
                                                          overwrite=overwrite))

    data_params = dict(
        graph_model = 'CSBM',
        classes = 2,
        n = 1000,
        n_per_class_trn = 400,
        K = 0.5,
        sigma = 1,
        avg_within_class_degree = 1.58 * 2,
        avg_between_class_degree = 0.37 * 2
    )

    model_params = dict(
        label="GCN",
        model="DenseGCN",
        n_filters=64,
        dropout=0.5,
        use_label_propagation=False,
    )

    train_params = dict(
        lr=1e-2,
        weight_decay=1e-3,
        patience=300,
        max_epochs=3000,
        inductive=True
    )

    verbosity_params = dict(
        display_steps = 100,
        debug_lvl = "info"
    )  

    other_params = dict(
        device = 0,
        allow_tf32 = False,
        sacred_metrics = True
    )

    seed = 1
    ex.path = get_exp_name(data_params, model_params, train_params, seed)


def set_debug_lvl(debug_lvl: str):
    if debug_lvl is not None and isinstance(debug_lvl, str):
        logger = logging.getLogger()
        if debug_lvl.lower() == "info":
            logger.setLevel(logging.INFO)
        if debug_lvl.lower() == "debug":
            logger.setLevel(logging.DEBUG)
        if debug_lvl.lower() == "critical":
            logger.setLevel(logging.CRITICAL)
        if debug_lvl.lower() == "error":
            logger.setLevel(logging.ERROR)


def log_configuration(data_params: Dict[str, Any], 
                      model_params: Dict[str, Any], 
                      train_params: Dict[str, Any], 
                      verbosity_params: Dict[str, Any], 
                      other_params: Dict[str, Any], seed: int, 
                      db_collection: Optional[str]) -> None:
    """Log (print) experiment configuration."""
    logging.info(f"Starting experiment {ex.path} with configuration:")
    logging.info(f"data_params: {data_params}")
    logging.info(f"model_params: {model_params}")
    logging.info(f"train_params: {train_params}")
    logging.info(f"verbosity_params: {verbosity_params}")
    logging.info(f"other_params: {other_params}")
    logging.info(f"seed: {seed}")
    logging.info(f"db_collection: {db_collection}")


def log_results(best_epoch: int, best_training_loss: float, 
                best_validation_loss: float, best_training_accuracy: float, 
                best_validation_accuracy: float, _run: Run):
    logging.info("Results of model with best lowest validation loss: ")
    logging.info(f"Best Epoch (1-based): {best_epoch}")
    logging.info(f"Training Loss: {best_training_loss:.4f}; "
                 f"Validation Loss: {best_validation_loss:.4f}")
    logging.info(f"Training Accuracy: {best_training_accuracy*100:.2f}; "
                 f"Validation Accuracy: {best_validation_accuracy*100:.2f}")
    if _run is not None:
        _run.log_scalar("best_epoch", best_epoch)
        _run.log_scalar("best_loss_trn", best_training_loss)
        _run.log_scalar("best_loss_val", best_validation_loss)
        _run.log_scalar("best_acc_trn", best_training_accuracy)
        _run.log_scalar("best_acc_val", best_validation_accuracy)
    

def configure_logging(verbosity_params: Dict[str, Any], 
                      other_params: Dict[str, Any], _run: Run) -> Run:
    """Return Run object if sacred metrics should be collected. """
    set_debug_lvl(verbosity_params["debug_lvl"])
    if not other_params["sacred_metrics"]:
        _run = None
    return _run


def configure_hardware(
    other_params: Dict[str, Any], seed: int
) -> Union[torch.device, str]:
    """Configure seed and computational hardware. Return calc. device."""
    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Note on Reproducability: Some GNNs based on PyTorch Geometric make heavy
    # use of non-deterministic scatter_add_() function 
    # (https://pytorch.org/docs/stable/notes/randomness.html)
    # for which no deterministic implementation exists. Hence, not all results
    # can be reproduced in a deterministic manner.

    # Hardware
    torch.backends.cuda.matmul.allow_tf32 = other_params["allow_tf32"]
    torch.backends.cudnn.allow_tf32 = other_params["allow_tf32"]

    device = other_params["device"]
    if not torch.cuda.is_available():
        assert device == "cpu", "CUDA is not availble, set device to 'cpu'"
    else:
        device = torch.device(f"cuda:{device}")
        logging.info(f"Currently on gpu device {device}")

    return device


@ex.automain
def run(data_params: Dict[str, Any], 
        model_params: Dict[str, Any], 
        train_params: Dict[str, Any], 
        verbosity_params: Dict[str, Any], 
        other_params: Dict[str, Any],
        seed: int, 
        db_collection: Optional[str], _run: Run):
    """ Run experiment with given configuration.

    _run: Run
        Used to log statistics using sacred.
    """
    log_configuration(data_params, model_params, train_params,
                      verbosity_params, other_params, seed, db_collection)
    _run = configure_logging(verbosity_params, other_params, _run)
    device = configure_hardware(other_params, seed)

    # Sample Graph
    graph_model = create_graph_model(data_params)
    X_np, A_np, y_np = graph_model.sample(data_params["n"], seed)
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    A = torch.tensor(A_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, device=device)
    split_trn, split_val = split(y_np, data_params)

    # Create Model
    model_params_trn = dict(**model_params, 
                            n_features=X_np.shape[1], 
                            n_classes=data_params["classes"])
    model = create_model(model_params_trn)
    if model is not None:
        model = model.to(device)
    lp = None
    if model_params["use_label_propagation"]:
        lp = LP(model_params["lp_layers"], 
                model_params["lp_alpha"], 
                data_params["classes"]).to(device)
    #logging.info(model)

    # Train Model
    if train_params["inductive"]:
        train = train_inductive
    else:
        train = train_transductive
    trn_tracker = train(model, lp, X, A, y, split_trn, split_val, train_params,
                        verbosity_params, _run)

    # Logging
    best_epoch = trn_tracker.get_best_epoch()
    training_loss = trn_tracker.get_training_loss()
    validation_loss = trn_tracker.get_validation_loss()
    training_accuracy = trn_tracker.get_training_accuracy()
    validation_accuracy = trn_tracker.get_validation_accuracy()
    best_training_loss = training_loss[best_epoch - 1]
    best_validation_loss = validation_loss[best_epoch - 1]
    best_training_accuracy = training_accuracy[best_epoch - 1]
    best_validation_accuracy = validation_accuracy[best_epoch - 1]
    log_results(best_epoch, best_training_loss, best_validation_loss, 
                best_training_accuracy, best_validation_accuracy, _run)
    return dict(
        training_loss = training_loss,
        validation_loss = validation_loss,
        training_accuracy = training_accuracy,
        validation_accuracy = validation_accuracy,
        training_epochs = trn_tracker.get_training_epochs(),
        best_epoch = best_epoch,
        best_training_loss = best_training_loss,
        best_validation_loss = best_validation_loss,
        best_training_accuracy = best_training_accuracy,
        best_validation_accuracy = best_validation_accuracy,
    )

