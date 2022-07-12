import numpy as np

class LocalAttack:
    """ Stores a graph & target node index. Provides method to manipulate 
    graph through local adversarial attacks.
    """
    def __init__(self, n_idx: int, X: np.ndarray, A: np.ndarray, 
                       y: np.ndarray, method="random") -> None:
        self.target = n_idx
        self.X = X
        self.A = A
        self.y = y
        self.pot_neighbours = self.prepare_attack(method)

    def prepare_attack(self, method) -> np.ndarray:
        """Return list of potential nodes to connect to depending on attack."""
        if method == "random":
            return self.random_edge_candidates()
        if method == "l2":
            return self.most_distant_edge_candidates()
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
        y, n_idx = self.y, self.target
        n = y.shape[0]
        if y[n_idx] == 0:
            pot_neighbours = np.arange(0, n)[y == 1]
        else:
            pot_neighbours = np.arange(0, n)[y == 0]
        X_diff = self.X[n_idx,:] - self.X[pot_neighbours,:]
        X_dist = np.linalg.norm(X_diff, ord=2, axis=1)
        idx_ascending = np.argsort(X_dist)
        idx_descending = np.flip(idx_ascending)
        return pot_neighbours[idx_descending]

    def add_adversarial_edge(self) -> int:
        """Add an adversarial edge to the stored graph.

        Connect target node with a node of different class not already in 
        its neighbourhood. Sorting of potential new neighbours defined by
        attack method in initialization.
        
        Return nodes index newly connected to target node or -1 if no 
        neighbours to connect to anymore.
        """
        A, n_idx = self.A, self.target
        for j in self.pot_neighbours:
            if A[n_idx,j] == 1:
                continue
            else:
                A[n_idx,j] = 1
                A[j,n_idx] = 1
                return j
        return -1
        #assert False, "Case adversarial edge to add not handeled."