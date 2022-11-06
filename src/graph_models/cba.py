import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import multivariate_normal

from src.graph_models.base_model import GraphGenerationModel


class CBA(GraphGenerationModel):
    """X, A ~ CBA(p, q, m, mu, cov, pi)

    Contextual Barabasi-Albert model with community structure for two classes.
    
    Affinities p and q will be set to the given average within or between class
    degree of nodes of an example real world graph.

    p ... inner-class affinity
    q ... inter-class affinity

    Args:
        n (int): Number of nodes
        m (int): Number of outgoing edges for each newly sampled node.
        avg_within_class_degree (float): Defines p.
        avg_between_class_degree (float): Defines q.
        K (float): Defines distance between means of the gaußians in 
            sigma-units.
        sigma (float): Standard deviation of the Gaußian distribution the
            node featuers are sampled from.
    m ... number of edges per iteration
    mu ... class dependent mean, must be of size len(pi) x d, check left to caller
    """
    def __init__(self, n: int, m: int, avg_within_class_degree: float, 
                 avg_between_class_degree: float, K: float, sigma: float,
                 **kwargs) -> None:
        self.p = avg_within_class_degree
        self.q = avg_between_class_degree
        self.m = m
        self.d = round(n / math.log(n)**2)
        self.mu = np.array([K*sigma / (2 * self.d**0.5) for i in range(self.d)], 
                           dtype=np.float32)
        self.cov = sigma**2 * np.identity(self.d, dtype=np.float32)

    def sample(self, n, seed=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample y~Bin(1/2) and X,A ~ CBA(n, p, q, m, mu, cov, pi). 

        y has shape [n,]. X has shape [n, d]. A has shape [n,n].

        Args:
            n (int): Number of nodes to sample
            seed (int, optional): Random seed. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: X, A, y
        """
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

        # Prepare A
        A = np.zeros((n,n))
        deg = np.ones(n,) # add each node with a virtual self-loop / 
                          # initial attractiveness of 1
        affinity = np.zeros(n,)

        for i in range(n):
            # Calculate Affinities 
            idx_same = y[:i] == y[i]
            idx_diff = y[:i] != y[i]
            affinity[:i][idx_same] = self.p
            affinity[:i][idx_diff] = self.q 
            if i > 0:
                # Calculate Preferential Attachment Probabilities
                p_draw = deg[:i] * affinity[:i]
                p_draw = p_draw / np.sum(p_draw)
                # Draw Nodes using PA
                draw = np.random.multinomial(self.m, p_draw) > 0
                deg[:i] += draw 
                deg[i] += np.sum(draw)
                A[:i,i] = draw
        A += A.T
        return X.astype(np.float32), A.astype(int), y
        
    
    def sample_conditional(self, n, X: np.ndarray, A: np.ndarray, y: np.ndarray) \
                                 -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample X', A', y' ~ D_n(X, A, y)

        Args:
            n (int): Number of inductively sampled nodes (currently only n=1 
                supported).
            X (np.ndarray): Existing nodes features.
            A (np.ndarray): Existing graph structure.
            y (np.ndarray): Existing graph labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: X, A, y
        """
        assert n == 1, "Only implemented for inductive sampling of a single node"

        # Sample y' | y:
        y_n = np.random.randint(0, 2, size=n)
        y_new = np.hstack((y, y_n))

        # Sample X' | y', X
        x_n = np.random.multivariate_normal((2*y_n - 1)*self.mu, self.cov, 
                                            size=n).astype(np.float32)
        X_new = np.vstack((X, x_n))

        # Sample A' | y', A
        deg = np.sum(A, axis=0) + 1
        old_n = len(y)
        # Calculate Affinities 
        idx_same = y_new[:old_n] == y_new[old_n]
        idx_diff = y_new[:old_n] != y_new[old_n]
        affinity = np.zeros(old_n+n,)
        affinity[:old_n][idx_same] = self.p
        affinity[:old_n][idx_diff] = self.q 
        # Calculate Preferential Attachment Probabilities
        p_draw = deg[:old_n] * affinity[:old_n]
        p_draw = p_draw / np.sum(p_draw)
        # Draw Nodes using PA
        draw = np.random.multinomial(self.m, p_draw) > 0
        A_tmp = np.vstack((A, draw))
        draw = np.hstack((draw, 0)) # new nodes has no connection to itself
        draw = draw.reshape((old_n+1, 1))
        A_new = np.hstack((A_tmp, draw))

        return X_new, A_new, y_new


    def loglikelihood(self, n_id, n_cls, X, A, y):
        """Return loglikelihood of node n_id being class n_cls given graph X,A,y.

        Method assumes n_id is last index in A, i.e. the last added node!
        
        Graph must be complete graph including n_id, likelihood is calculated
        as if the whole graph except label of n_id-th node is known.
        """
        assert n_id == len(y) - 1

        likelihood = math.log(multivariate_normal.pdf(X[n_id,:], 
                                                      mean=(2*n_cls - 1)*self.mu, 
                                                      cov=self.cov))
        likelihood += self.structure_loglikelihood(n_id, n_cls, A, y)
        
        return likelihood

    def structure_loglikelihood(self, n_id, n_cls, A, y):
        """Return structure log-likelihood of node n_id being of class n_cls.

        Method assumes n_id is last index in A, i.e. the last added node!
        
        Graph X,A,y must be complete graph including n_id, likelihood is 
        calculated as if the whole graph except label of n_id-th node is known.
        """
        assert n_id == len(y) - 1
        #### Calculate Drawing Probabilities
        deg = np.sum(A, axis=0) + 1
        # Correct for new edges from n_id
        draw = A[n_id,:] > 0
        deg = deg - draw
        # Calculate Affinities 
        idx_same = y[:n_id] == n_cls
        idx_diff = y[:n_id] != n_cls
        affinity = np.zeros(n_id,)
        affinity[:n_id][idx_same] = self.p
        affinity[:n_id][idx_diff] = self.q 
        # Calculate Preferential Attachment Probabilities
        p_draw = deg[:n_id] * affinity[:n_id]
        p_draw = p_draw / np.sum(p_draw)

        ##### Calculate Structure Likelhood
        likelihood = 0
        for i in range(len(y)-1):
            if A[n_id, i] == 1:
                likelihood += math.log(p_draw[i])
            else:
                likelihood += math.log(1-p_draw[i])
        return likelihood

    def feature_separability(self, X, y, ids=None) -> Tuple[int, int]:
        """Check (bayes) feature separability of graph X, y given (data-gen) 
        model.
        
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

    def structure_separability(self, A, y, ids=None) -> Tuple[int, int]:
        """Return tuple: #separable, #non-separable.
        
        Method assumes ids has size 1 and is last index in A, i.e. the last 
        added node!
        """
        assert len(ids) == 1 and ids[0] == len(y) - 1
        n_id = ids[0]
        n_corr = 0
        n_wrong = 0
        likelihood_corr = self.structure_loglikelihood(n_id, y[n_id], A, y)
        likelihood_wrong = self.structure_loglikelihood(n_id, -(y[n_id]-1), A, y)
        if likelihood_corr > likelihood_wrong:
            n_corr += 1
        else:
            n_wrong += 1
        return n_corr, n_wrong

    def likelihood_separability(self, X, A, y, ids=None) -> Tuple[int, int]:
        """Return tuple: #separable, #non-separable.
        
        Method assumes ids has size 1 and is last index in A, i.e. the last 
        added node!
        """
        assert len(ids) == 1 and ids[0] == len(y) - 1
        n_id = ids[0]
        n_corr = 0
        n_wrong = 0
        likelihood_corr = self.loglikelihood(n_id, y[n_id], X, A, y)
        likelihood_wrong = self.loglikelihood(n_id, -(y[n_id]-1), X, A, y)
        if likelihood_corr > likelihood_wrong:
            n_corr += 1
        else:
            n_wrong += 1
        return n_corr, n_wrong