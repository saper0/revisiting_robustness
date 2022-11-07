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
from src.models import create_model, LP
from src.train import train_inductive, train_transductive, test

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
                 attack_params: Dict[str, Any],
                 seed) -> str:
    K = data_params["K"]
    exp_name = "robustness_" + data_params["graph_model"] + f"_K{K:.1f}_"
    if train_params["inductive"]:
        exp_name += "inductive" + "_" 
    else:
        exp_name += "transductive" + "_"
    exp_name += model_params["label"] + "_" + attack_params["attack"]
    exp_name += f"_seed{seed}"
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
        avg_between_class_degree = 0.37 * 2,
        inductive_samples = 1000,
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

    attack_params = dict(
        attack = "l2"
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
    ex.path = get_exp_name(data_params, model_params, train_params, 
                           attack_params, seed)


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
                      train_params: Dict[str, Any], attack_params: Dict[str, Any],
                      verbosity_params: Dict[str, Any], 
                      other_params: Dict[str, Any], seed: int, 
                      db_collection: Optional[str]) -> None:
    """Log (print) experiment configuration."""
    logging.info(f"Starting experiment {ex.path} with configuration:")
    logging.info(f"data_params: {data_params}")
    logging.info(f"model_params: {model_params}")
    logging.info(f"train_params: {train_params}")
    logging.info(f"attack_params: {attack_params}")
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


def log_robust_statistics(c_bayes_higher_robust, c_bayes_gnn_equal_robust,
                            c_gnn_higher_robust):
    logging.info(f"Robustness Statistics:")
    logging.info(f"BC more robust than GNN: {c_bayes_higher_robust:.1f}")
    logging.info(f"BC & GNN equal robustness: {c_bayes_gnn_equal_robust:.1f}")
    logging.info(f"BC less robust than GNN: {c_gnn_higher_robust:.1f}")


def log_deg_dict(name_deg_dict1: str, deg_dict1: Dict[int, int],  
                   name_deg_dict2: str, deg_dict2: Dict[int, int]):
    """Log Degree-Dependent Robustness of BC and GNN."""
    max_deg = max([max([int(deg) for deg in deg_dict1.keys()]), 
                   max([int(deg) for deg in deg_dict2.keys()])])
    ordered_avg_dict1 = [deg_dict1[str(i)] if str(i) in deg_dict1 else -1 for i in range(max_deg+1)]
    ordered_avg_dict2 = [deg_dict2[str(i)] if str(i) in deg_dict2 else -1 for i in range(max_deg+1)]
    for deg in range(max_deg+1):
        logging.info(f"Degree {deg}: {name_deg_dict1}: {ordered_avg_dict1[deg]:.2f}; "
                     f"{name_deg_dict2}: {ordered_avg_dict2[deg]:.2f}; ")


def log_wrt_bayes_dicts(gnn_wrt_bayes_robust, bayes_robust_when_both,
                        gnn_robust_when_both):
    """Log robustness w.r.t. bayes classifier results."""
    max_deg = max([max([int(deg) for deg in gnn_wrt_bayes_robust.keys()]), 
                   max([int(deg) for deg in bayes_robust_when_both.keys()]),
                   max([int(deg) for deg in gnn_robust_when_both.keys()])])
    ordered_gnn_wrt_bayes_robust = [gnn_wrt_bayes_robust[str(i)] if str(i) in gnn_wrt_bayes_robust else -1 for i in range(max_deg+1)]
    ordered_bayes_robust_when_both = [bayes_robust_when_both[str(i)] if str(i) in bayes_robust_when_both else -1 for i in range(max_deg+1)]
    ordered_gnn_robust_when_both = [gnn_robust_when_both[str(i)] if str(i) in gnn_robust_when_both else -1 for i in range(max_deg+1)]
    for deg in range(max_deg+1):
        logging.info(
            f"Degree {deg}: <GNN wrt BC robust>: {ordered_gnn_wrt_bayes_robust[deg]:.2f}/"
            f"{ordered_bayes_robust_when_both[deg]:.2f}. <GNN in wrt BC setting>: "
            f"{ordered_gnn_robust_when_both[deg]:.2f}"
        )


@ex.automain
def run(data_params: Dict[str, Any], 
        model_params: Dict[str, Any], 
        train_params: Dict[str, Any], 
        attack_params: Dict[str, Any],
        verbosity_params: Dict[str, Any], 
        other_params: Dict[str, Any],
        seed: int, 
        db_collection: Optional[str], _run: Run):
    """ Run experiment with given configuration.

    _run: Run
        Used to log statistics using sacred.
    """
    log_configuration(data_params, model_params, train_params, attack_params,
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
    
    #not_trained = True
    #while not_trained:
    model = create_model(model_params_trn)
    if model is not None:
        model = model.to(device)
    label_prop = None
    if model_params["use_label_propagation"]:
        if model_params["lp_use_clamping"]:
            post_step = lambda y: y.clamp_(0, 1)
        else:
            post_step = lambda y: y
        label_prop = LP(model_params["lp_layers"], 
                        model_params["lp_alpha"], 
                        data_params["classes"],
                        post_step).to(device)
    #logging.info(model)

    # Train Model
    if train_params["inductive"]:
        train = train_inductive
    else:
        train = train_transductive
    if model is not None:
        # Train model independently from label propagation.
        lp_in_training = None
    else:
        lp_in_training = label_prop
    train_tracker = train(model, lp_in_training, X, A, y, split_trn, split_val, 
                        train_params, verbosity_params, _run)
    #if train_tracker.get_best_epoch() == 1 and lp_in_training is None:
    #    logging.info("Model did not train, re-initialize model.")
    #else:
    #    not_trained = False

    if label_prop is not None:
        logging.info("Testing Trained Model + Label Propagation:")
        test_tracker = test(model, label_prop, X, A, y, split_trn, split_val, _run)
    else:
        test_tracker = train_tracker

    # Robustness Evaluation
    surrogate_model = None
    if attack_params["attack"] == "nettack" or \
        attack_params["attack"] == "nettack_power_law_test":
        # Train surrogate model
        surrogate_model_params = dict(**attack_params["surrogate_model_params"],
                                      n_features=X_np.shape[1], 
                                      n_classes=data_params["classes"])
        surrogate_train_params = attack_params["surrogate_train_params"]
        surrogate_model = create_model(surrogate_model_params).to(device)
        train(surrogate_model, None, X, A, y, split_trn, split_val, 
              surrogate_train_params, verbosity_params, _run)
        surrogate_model.eval()
    results_dict = evaluate_robustness(model, 
                                       label_prop,
                                       graph_model, 
                                       X_np, A_np, y_np,
                                       data_params["inductive_samples"], 
                                       attack_params,
                                       surrogate_model,
                                       device)

    # (Optional) Logging of Robusntess Statistics
    log_prediction_statistics(**results_dict["prediction_statistics"])
    robustness_statistics = results_dict["robustness_statistics"]
    log_robust_statistics(
        c_bayes_gnn_equal_robust=robustness_statistics["c_bayes_gnn_equal_robust"],
        c_bayes_higher_robust=robustness_statistics["c_bayes_higher_robust"],
        c_gnn_higher_robust=robustness_statistics["c_gnn_higher_robust"]
    )
    log_deg_dict("<BC robust>", robustness_statistics["avg_bayes_robust"], 
                 "<GNN robust>", robustness_statistics["avg_gnn_robust"])
    log_deg_dict("Max(BC robust)", robustness_statistics["max_bayes_robust"], 
                 "Max(GNN robust)", robustness_statistics["max_gnn_robust"])
    log_deg_dict("Median(BC robust)", robustness_statistics["med_bayes_robust"], 
                 "Median(GNN robust)", robustness_statistics["med_gnn_robust"])
    log_wrt_bayes_dicts(robustness_statistics["avg_gnn_wrt_bayes_robust"], 
                        robustness_statistics["avg_bayes_robust_when_both"], 
                        robustness_statistics["avg_gnn_robust_when_both"])
    
    used_epoch = test_tracker.get_best_epoch() - 1
    return dict(
        prediction_statistics = results_dict["prediction_statistics"],
        robustness_statistics = results_dict["robustness_statistics"],
        training_loss = train_tracker.get_training_loss(),
        validation_loss = train_tracker.get_validation_loss(),
        training_accuracy = train_tracker.get_training_accuracy(),
        validation_accuracy = train_tracker.get_validation_accuracy(),
        final_training_loss = test_tracker.get_training_loss()[used_epoch],
        final_training_accuracy = test_tracker.get_training_accuracy()[used_epoch],
        final_validation_loss = test_tracker.get_validation_loss()[used_epoch],
        final_validation_accuracy = test_tracker.get_validation_accuracy()[used_epoch]
    )

