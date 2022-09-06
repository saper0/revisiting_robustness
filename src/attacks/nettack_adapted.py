from importlib.util import module_for_loader
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torchtyping import TensorType

from src.attacks.base_attack import LocalAttack
from src.models.lp import LP


class NettackAdapted(LocalAttack):
    """ Stores a graph & target node index. Provides method to manipulate 
    graph through local adversarial attacks.
    """
    def __init__(
        self, attack: str, n_idx: int, X: np.ndarray, A: np.ndarray, 
        y: np.ndarray, model: Optional[nn.Module], label_prop: Optional[LP], 
        device: Union[torch.device, str]
    ) -> None:
        assert model is not None or label_prop is None
        assert device is not None
        self.target = n_idx
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.A = torch.tensor(A, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, device=device)
        self.model = model
        if self.model is not None:
            self.model.eval()
        self.label_prop = label_prop
        n = A.shape[0]
        self.potential_neighbours = np.array([True for i in range(n)])
        self.potential_neighbours[self.target] = False
        self.node_ids = np.array([i for i in range(n)])
        self.node_ids_wo_target = self.node_ids[self.potential_neighbours]
        self.attack = attack

    def prepare_attack(self, attack) -> np.ndarray:
        """Return list of potential nodes to connect to depending on attack."""
        if attack == "nettack-adapted":
            return self.get_worst_edge(direct_attack=True)
        assert False, "Given attack method not implemented."

    def get_worst_edge(self, direct_attack=True) -> Tuple[int, int]:
        """Return edge which results in maximal attack loss.

        Attack loss is defined as distance between the logits of the 
        "strongest" wrong class and the true class. Strongest means the wrong
        class with maximal logits.

        Args:
            direct_attack (bool, optional): Defaults to True.

        Returns:
            Tuple[int, int]: Node-index-tuple of newly added or removed edge 
                or None if no perturbation possible anymore.
        """
        assert direct_attack
        assert self.attack == "nettack-adapted"

        # Performance on unperturbed graph (ignore ln because monotonic)
        logits = self.forward()
        Z_true = logits[self.target, self.y[self.target]]
        c_other = [c for c in range(logits.size(1)) if c != self.y[self.target]]

        max_loss = None
        adv_n_idx = None
        for n_idx in self.node_ids[self.potential_neighbours]:
            self.add_or_remove_edge(n_idx, self.target)
            logits = self.forward()
            Z_max_other = torch.max(logits[self.target, c_other])
            loss = Z_max_other - Z_true
            if max_loss is None or max_loss < loss:
                max_loss = loss
                adv_n_idx = n_idx
            self.add_or_remove_edge(n_idx, self.target)

        self.add_or_remove_edge(adv_n_idx, self.target, verbose=False)
        self.potential_neighbours[adv_n_idx] = False # don't add and in later iteration remove that edge
        if max_loss is not None:
            return adv_n_idx, self.target
        else:
            return None

    def forward(self) -> TensorType["n", "c"]:
        """Forward currently stored graph through model+LP and return logits.
        """
        if self.model is not None:
            logits = self.model(self.X, self.A)
            normalize = True
        else:
            logits = None
            normalize = False
        if self.label_prop is not None:
            trn_nodes = self.node_ids_wo_target
            logits = self.label_prop.smooth(
                logits, self.y[trn_nodes], trn_nodes, self.A, normalize)
        return logits

    def add_or_remove_edge(self, u: int, v: int, verbose: bool=False) -> None:
        """Given (u,v), add edge to graph if not exists, otherwise remove."""
        if self.A[u, v] == 1:
            self.A[u, v] = 0
            self.A[v, u] = 0
            if verbose:
                print(f"Removed edge ({u},{v})")
        else:
            self.A[u, v] = 1
            self.A[v, u] = 1
            if verbose:
                print(f"Added edge ({u},{v})")

    def create_adversarial_pert(self) -> Tuple[int, int]:
        """Add an adversarial edge to the stored graph.

        Connect target node with a node of different class not already in 
        its neighbourhood. Sorting of potential new neighbours defined by
        attack method in initialization. If attack method "random" chooses
        a random node of different class, if attack method "l2" chooses most
        distant node w.r.t. node features.

        Returns:
            Tuple[int, int]: Node-index-tuple of newly added or removed edge 
                or None if no perturbation possible anymore.
        """
        return self.get_worst_edge(direct_attack=True)