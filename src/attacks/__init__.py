from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.attacks.simple_attack import SimpleAttack
from src.attacks.nettack import Nettack
from src.attacks.nettack_adapted import NettackAdapted
from src.attacks.sga import SGA
from src.attacks.rbcd import RBCDWrapper
from src.models.lp import LP

ATTACK_TYPE = [SimpleAttack, Nettack, NettackAdapted, RBCDWrapper, SGA]

def create_attack(target_idx: int, X: np.ndarray, A: np.ndarray, y: np.ndarray,
                  hyperparams: Dict[str, Any], 
                  surrogate_model: Optional[nn.Module]=None,
                  model: Optional[nn.Module]=None, 
                  label_prop: Optional[LP]=None,
                  device: Union[torch.device, str]=None) -> ATTACK_TYPE:
    """Initialize a local attack on a target node.

    Args:
        target_idx (int): Target node index.
        X (np.ndarray): Feature matrix of graph.
        A (np.ndarray): Adjacency matrix.
        y (np.ndarray): Node labels.
        hyperparams (Dict[str, Any]): Parameters of local attack. Must include
            key "attack" specifying the name of the attack to create.
        surrogate_model: (Optional[nn.Module]): Only relevant if nettack is 
            used. Surrogate model to attack to get perturbed adjacency matrix.
        model (Optional[nn.Module]): Only used for nettack-adapted.
        label_prop (Optional[LP]): Only used for nettack-adapted.
        device (Union[torch.device, str]): Only used for nettack-adapted.
    Raises:
        ValueError: If specified attack not found.

    Returns:
        ATTACK_TYPE: Initialized locl attack for a target node.
    """
    if hyperparams["attack"] == "random" or hyperparams["attack"] == "l2" \
        or hyperparams["attack"] == "l2-weak":
        return SimpleAttack(target_idx, X, A, y, hyperparams["attack"])
    if hyperparams["attack"] == "nettack":
        power_law_test = False
        if "power_law_test" in hyperparams:
            power_law_test = hyperparams["power_law_test"]
        return Nettack(target_idx, X, A, y, surrogate_model, power_law_test)
    if hyperparams["attack"] == "nettack_power_law_test":
        power_law_test = True
        return Nettack(target_idx, X, A, y, surrogate_model, power_law_test)
    if hyperparams["attack"] == "SGA":
        return SGA(target_idx, X, A, y, surrogate_model,
                   n_perturbations=hyperparams["max_robustness"],
                   device=device)
    if hyperparams["attack"] == "nettack-adapted":
        return NettackAdapted(hyperparams["attack"], target_idx, X, A, y, 
                              model, label_prop, device)
    if hyperparams["attack"] in ["PRBCD", "GRBCD"]:
        return RBCDWrapper(hyperparams["attack"], target_idx, X, A, y,  model)
    raise ValueError("Specified attack not found.")

__all__ = [SimpleAttack, Nettack, NettackAdapted, RBCDWrapper, SGA,
           ATTACK_TYPE, create_attack]