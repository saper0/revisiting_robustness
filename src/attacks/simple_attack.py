from importlib.util import module_for_loader
from typing import Tuple

import numpy as np

from src.attacks.base_attack import LocalAttack


class SimpleAttack(LocalAttack):
    """ Stores a graph & target node index. Provides method to manipulate 
    graph through local adversarial attacks.
    """
    def __init__(self, n_idx: int, X: np.ndarray, A: np.ndarray, 
                       y: np.ndarray, attack="random") -> None:
        self.target = n_idx
        self.X = np.copy(X)
        self.A = np.copy(A)
        self.y = np.copy(y)
        self.pot_neighbours = self.prepare_attack(attack)

    def prepare_attack(self, attack) -> np.ndarray:
        """Return list of potential nodes to connect to depending on attack."""
        if attack == "random":
            return self.random_edge_candidates()
        if attack == "l2":
            return self.most_distant_edge_candidates()
        if attack == "l2-weak":
            return self.most_closest_edge_candidates()
        assert False, "Given attack method not implemented."

    def random_edge_candidates(self) -> np.ndarray:
        """Return list of nodes of different class than target in random order.
        
        Prepares to connect target node with a random node. Like local DICE 
        but without removing random nodes."""
        y, n_idx = self.y, self.target
        n = y.shape[0]
        if y[n_idx] == 0:
            pot_neighbours = np.arange(0, n)[y == 1]
        else:
            pot_neighbours = np.arange(0, n)[y == 0]
        np.random.shuffle(pot_neighbours)
        return pot_neighbours

    def most_distant_edge_candidates(self) -> np.ndarray:
        """Return list of nodes of different class in order of decreasing 
        l2-distance."""
        return self.l2_edge_candidates(order="descending")

    def most_closest_edge_candidates(self) -> np.ndarray:
        """Return list of nodes of different class in order of increasing 
        l2-distance."""
        return self.l2_edge_candidates(order="ascending")

    def l2_edge_candidates(self, order="descending") -> np.ndarray:
        """Return list of nodes of different class in order of descending 
        (decreasing) or ascending (increasing) l2-distance."""
        y, n_idx = self.y, self.target
        n = y.shape[0]
        if y[n_idx] == 0:
            pot_neighbours = np.arange(0, n)[y == 1]
        else:
            pot_neighbours = np.arange(0, n)[y == 0]
        X_diff = self.X[n_idx,:] - self.X[pot_neighbours,:]
        X_dist = np.linalg.norm(X_diff, ord=2, axis=1)
        idx_ascending = np.argsort(X_dist)
        if order == "ascending":
            return pot_neighbours[idx_ascending]
        if order == "descending":
            idx_descending = np.flip(idx_ascending)
            return pot_neighbours[idx_descending]
        raise ValueError(f"Specified l2 candidate order must be descending or "
                         f"ascending but {order} given.")

    def create_adversarial_pert(self) -> Tuple[int, int]:
        """Add an adversarial edge to the stored graph.

        Connect target node with a node of different class not already in 
        its neighbourhood. Sorting of potential new neighbours defined by
        attack method in initialization. If attack method "random" chooses
        a random node of different class, if attack method "l2" chooses most
        distant node w.r.t. node features.

        Returns:
            Tuple[int, int]: Node-index-tuple of newly added edge or None if no
                perturbation possible anymore.
        """
        A, n_idx = self.A, self.target
        for j in self.pot_neighbours:
            if A[n_idx,j] == 1:
                continue
            else:
                A[n_idx,j] = 1
                A[j,n_idx] = 1
                return j, n_idx
        return None