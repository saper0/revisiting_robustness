import math
from typing import Tuple

import numpy as np
from scipy.stats import multivariate_normal
import torch
from torchtyping import TensorType

class CSBM:
    """X, A ~ CSBM(n, p, q, mu, cov)"""
    def __init__(self, p, q, mu, cov):
        self.p = p
        self.q = q
        self.mu = mu
        self.cov = cov
        self.d = len(mu)

    def sample(self, n, seed = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample y~Bin(1/2) and X,A ~ CSBM(n, p, q, mu, cov). 
        
        Return X: np.ndarray, 
               A: np.ndarray, 
               y: np.ndarray."""
        np.random.seed(seed)
        # Sample y
        y = np.random.randint(0,2,size=n)
        n_class1 = sum(y)
        n_class0 = len(y) - n_class1 
        # Sample X|y
        X_0 = np.random.multivariate_normal(-self.mu, self.cov, n_class0).astype(np.float32)
        X_1 = np.random.multivariate_normal(self.mu, self.cov, n_class1).astype(np.float32)
        X = np.zeros((n,self.d))
        X[y == 0, :] = X_0
        X[y == 1, :] = X_1
        #print(sum(torch.eq(torch.tensor(y), X[:,0] > 0)))
        # Sample A|y
        edge_prob = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1, n):
                if y[i] == y[j]:
                    edge_prob[i,j] = self.p
                else:
                    edge_prob[i,j] = self.q
        A = np.random.binomial(1, edge_prob)
        A += A.T
        return X, A, y

    def sample_conditional(self, n, X: np.ndarray, A: np.ndarray, y: np.ndarray) \
                                 -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample X', A', y' ~ D_n(X, A, y)"""
        assert n == 1, "Only implemented for inductive sampling of a single node"

        # Sample y' | y:
        y_n = np.random.randint(0, 2, size=n)
        y_new = np.hstack((y, y_n))

        # Sample X' | y', X
        x_n = np.random.multivariate_normal((2*y_n - 1)*self.mu, self.cov, size=n).astype(np.float32)
        X_new = np.vstack((X, x_n))

        # Sample A' | y', A
        old_n = len(y)
        edge_prob = np.zeros((n,old_n+n))
        for i in range(old_n):
            for j in range(n):
                if y_new[i] == y_new[old_n+j]:
                    edge_prob[j,i] = self.p
                else:
                    edge_prob[j,i] = self.q
        a_n = np.random.binomial(1, edge_prob)
        A_tmp = np.vstack((A, a_n[0, :-1]))
        A_new = np.hstack((A_tmp, a_n.T))

        return X_new, A_new, y_new


    def likelihood(self, n_id, n_cls, X, A, y):
        """Return likelihood of node n_id being class n_cls given graph X,A,y.
        
        Graph must be complete graph including n_id, likelihood is calculated
        as if the whole graph except label of n_id-th node is known.
        """
        likelihood = multivariate_normal.pdf(X[n_id,:], 
                                             mean=(2*n_cls - 1)*self.mu, 
                                             cov=self.cov)
        for i in range(len(y)):
            if i != n_id:
                if y[i] == n_cls:
                    likelihood *= self.p if A[n_id, i] == 1 else 1-self.p
                else:
                    likelihood *= self.q if A[n_id, i] == 1 else 1-self.q
        return likelihood

    def loglikelihood(self, n_id, n_cls, X, A, y):
        """Return loglikelihood of node n_id being class n_cls given graph X,A,y.
        
        Graph must be complete graph including n_id, likelihood is calculated
        as if the whole graph except label of n_id-th node is known.
        """
        likelihood = math.log(multivariate_normal.pdf(X[n_id,:], 
                                                      mean=(2*n_cls - 1)*self.mu, 
                                                      cov=self.cov))
        likelihood += self.structure_loglikelihood(n_id, n_cls, A, y)
        
        return likelihood

    def structure_loglikelihood(self, n_id, n_cls, A, y):
        """Return structure log-likelihood of node n_id being of class n_cls.
        
        Graph X,A,y must be complete graph including n_id, likelihood is 
        calculated as if the whole graph except label of n_id-th node is known.
        """
        likelihood = 0
        for i in range(len(y)):
            if i != n_id:
                if y[i] == n_cls:
                    likelihood += math.log(self.p) if A[n_id, i] == 1 \
                                                   else math.log(1-self.p)
                else: 
                    likelihood += math.log(self.q) if A[n_id, i] == 1 \
                                                   else math.log(1-self.q)
        return likelihood

    def feature_separability(self, X, y, ids=None):
        """Check (bayes) feature separability of graph X, A, y given CSBM.
        
        Optionally only check for nodes in ids.
        
        Return tuple: #separable, #non-separable."""
        n_corr = 0
        n_wrong = 0
        if ids is None:
            ids = [i for i in range(len(y))]
        for i in ids:
            density_corr = multivariate_normal.pdf(X[i,:], 
                                                   mean=(2*y[i] - 1)*self.mu, 
                                                   cov=self.cov)
            density_wrong = multivariate_normal.pdf(X[i,:], 
                                                    mean=-(2*y[i] - 1)*self.mu, 
                                                    cov=self.cov)
            if density_corr > density_wrong:
                n_corr += 1
            else:
                n_wrong += 1
        return n_corr, n_wrong

    def structure_separability(self, A, y, ids=None):
        """Return tuple: #separable, #non-separable."""
        # Check how much nodes are correct w.r.t. structure likelihood
        n_corr = 0
        n_wrong = 0
        if ids is None:
            ids = [i for i in range(len(y))]
        for i in ids:
            likelihood_corr = self.structure_loglikelihood(i, y[i], A, y)
            likelihood_wrong = self.structure_loglikelihood(i, -(y[i]-1), A, y)
            if likelihood_corr > likelihood_wrong:
                n_corr += 1
            else:
                n_wrong += 1
        return n_corr, n_wrong

    def likelihood_separability(self, X, A, y, ids=None):
        """Return tuple: #separable, #non-separable."""
        # Check how much nodes are correct w.r.t. likelihood
        n_corr = 0
        n_wrong = 0
        if ids is None:
            ids = [i for i in range(len(y))]
        for i in ids:
            likelihood_corr = self.loglikelihood(i, y[i], X, A, y)
            likelihood_wrong = self.loglikelihood(i, -(y[i]-1), X, A, y)
            if likelihood_corr > likelihood_wrong:
                n_corr += 1
            else:
                n_wrong += 1
        return n_corr, n_wrong

    def check_separabilities(self, X, A, y, ids=None):
        """Check separability of graph X,A,y. 
        
        Optionally only check for nodes in ids."""
        print(f"Feature Separability:")
        n_corr, n_wrong = self.feature_separability(X, y, ids)
        print(f"n_corr: {n_corr}")
        print(f"n_wrong: {n_wrong}")
        print(f"Structure Separability:")
        n_corr, n_wrong = self.structure_separability(A, y, ids)
        print(f"n_corr: {n_corr}")
        print(f"n_wrong: {n_wrong}")
        print(f"Likelihood Separability:")
        n_corr, n_wrong = self.likelihood_separability(X, A, y, ids)
        print(f"n_corr: {n_corr}")
        print(f"n_wrong: {n_wrong}")


def get_sbm_model(n, avg_intra_degree, avg_inter_degree, K=0.5, sigma=0.1) \
                                                                    -> CSBM:
    """
    Return correctly parametrized CSBM class. 
    avg_intra_degree = intra_edges_per_node * 2
    K ... Defines distance between means of the gaußians in sigma-units
    """
    p = avg_intra_degree * 2 / (n - 1)
    q = avg_inter_degree * 2 / (n - 1)
    d = round(n / math.log(n)**2)
    mu = np.array([K*sigma / (2 * d**0.5) for i in range(d)], dtype=np.float32)
    cov = sigma**2 * np.identity(d, dtype=np.float32)
    return CSBM(p, q, mu, cov)


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