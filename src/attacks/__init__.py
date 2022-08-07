from typing import Any, Dict

import numpy as np

from src.attacks.simple_attack import SimpleAttack

ATTACK_TYPE = SimpleAttack

def create_attack(target_idx: int, X: np.ndarray, A: np.ndarray, y: np.ndarray,
                  hyperparams: Dict[str, Any]) -> ATTACK_TYPE:
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
    if hyperparams["attack"] == "random" or hyperparams["attack"] == "l2":
        return SimpleAttack(target_idx, X, A, y, **hyperparams)
    raise ValueError("Specified attack not found.")

__all__ = [SimpleAttack, ATTACK_TYPE, create_attack]