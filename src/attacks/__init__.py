from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from src.attacks.simple_attack import SimpleAttack
from src.attacks.nettack import Nettack
from src.attacks.nettack_adapted import NettackAdapted
from src.models.lp import LP

ATTACK_TYPE = [SimpleAttack, Nettack, NettackAdapted]

def create_attack(target_idx: int, X: np.ndarray, A: np.ndarray, y: np.ndarray,
                  hyperparams: Dict[str, Any], model: Optional[nn.Module]=None, 
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

    Raises:
        ValueError: If specified attack not found.

    Returns:
        ATTACK_TYPE: Initialized locl attack for a target node.
    """
    if hyperparams["attack"] == "random" or hyperparams["attack"] == "l2" \
        or hyperparams["attack"] == "l2-weak":
        return SimpleAttack(target_idx, X, A, y, **hyperparams)
    if hyperparams["attack"] == "nettack":
        return Nettack(target_idx, X, A, y, model)
    if hyperparams["attack"] == "nettack-adapted":
        return NettackAdapted(hyperparams["attack"], target_idx, X, A, y, 
                              model, label_prop, device)
    raise ValueError("Specified attack not found.")

__all__ = [SimpleAttack, Nettack, NettackAdapted, ATTACK_TYPE, create_attack]