import logging
from typing import Any, Dict, Union, Optional

import numpy as np
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.run import Run
import torch

from src.data import split
from src.eval import evaluate_robustness
from src.graph_models import create_graph_model
from src.models import create_model
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
        n_per_class_trn = 250,
        K = 0.5,
        sigma = 0.1,
        avg_within_class_degree = 1.5 * 2,
        avg_between_class_degree = 0.5 * 2,
        inductive_samples = 1000,
    )

    model_params = dict(
        label="GCN",
        model="DenseGCN",
        n_filters=64,
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
    ex.path = "name_of_exp"


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


def log_configuration(data_params: Dict[str, Any], model_params: Dict[str, Any], 
                      train_params: Dict[str, Any], verbosity_params: Dict[str, Any], 
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


def log_prediction_statistics(c_acc_bayes, c_acc_gnn, c_acc_bayes_structure,
                              c_acc_bayes_feature, c_acc_bayes_gnn, 
                              c_acc_bayes_not_gnn, c_acc_gnn_not_bayes):
    logging.info(f"Prediction Statistics:")
    logging.info(f"Count BC: {c_acc_bayes:.1f} GNN: {c_acc_gnn:.1f}")
    logging.info(f"Count Structure BC: {c_acc_bayes_structure:.1f} "
                 f"Feature BC: {c_acc_bayes_feature:.1f}")
    logging.info(f"Count BC and GNN: {c_acc_bayes_gnn:.1f}")
    logging.info(f"Count BC not GNN: {c_acc_bayes_not_gnn:.1f} "
                 f"GNN not BC: {c_acc_gnn_not_bayes:.1f}")


@ex.automain
def run(data_params: Dict[str, Any], 
        model_params: Dict[str, Any], 
        train_params: Dict[str, Any], 
        verbosity_params: Dict[str, Any], 
        other_params: Dict[str, Any],
        seed: int, db_collection: Optional[str], _run: Run):
    """ Run experiment with given configuration.

    _run: Run
        Used to log statistics using sacred.
    """
    log_configuration(data_params, model_params, train_params, verbosity_params, 
                      other_params, seed, db_collection)
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
    model = create_model(model_params_trn).to(device)
    #logging.info(model)

    # Train Model
    if train_params["inductive"]:
        train = train_inductive
    else:
        train = train_transductive
    statistics = train(model, X, A, y, split_trn, split_val, train_params,
                       verbosity_params, _run)

    # Robustness Evaluation
    prediction_stats = evaluate_robustness(model, 
                                           graph_model, 
                                           X_np, A_np, y_np,
                                           data_params["inductive_samples"], 
                                           device)

    log_prediction_statistics(*prediction_stats[0])