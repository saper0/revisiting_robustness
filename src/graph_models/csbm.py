import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import multivariate_normal

from src.graph_models.base_model import GraphGenerationModel

class CSBM(GraphGenerationModel):
    """A contextual stochastic block model for the two-class setting. """
    def __init__(self, n: int, avg_within_class_degree: float, 
                 avg_between_class_degree: float, K: float, sigma: float,
                 **kwargs) -> None:
        """Create correct parameter settings for a CSBM. 

        Sets within-class probability p and between-class probability q in a 
        way such that for the given n nodes in expectation avg_within_class_degree
        and avg_between_class_degree is realized.

        Args:
            n (int): Number of nodes
            avg_within_class_degree (float): Expected number of edges a node
                has to other nodes of his class.
            avg_between_class_degree (float): Expected number of edges a noe
                has to other nodes of different class.
            K (float): Defines distance between means of the gaußians in 
                sigma-units.
            sigma (float): Standard deviation of the Gaußian distribution the
                node featuers are sampled from.
        """
        assert kwargs["classes"] == 2, "Only two-community CSBM implemented"
        self.p = avg_within_class_degree * 2 / (n - 1)
        self.q = avg_between_class_degree * 2 / (n - 1)
        self.d = round(n / math.log(n)**2)
        self.mu = np.array([K*sigma / (2 * self.d**0.5) for i in range(self.d)], 
                           dtype=np.float32)
        self.cov = sigma**2 * np.identity(self.d, dtype=np.float32)

    def sample(self, n: int, seed = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample y~Bin(1/2) and X,A ~ CSBM(n, p, q, mu, cov). 

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

    def sample_conditional(self, n: int, X: np.ndarray, A: np.ndarray, 
                           y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def likelihood(self, n_id: int, n_cls: int, X: np.ndarray, A: np.ndarray, 
                   y: np.ndarray) -> float:
        """Calculate likelihood of a given node being of a certain class.
        
        Likelihood calculation is based on a given realized graph structure
        with the assumption the graph was generated using this CSBM model. 

        Args:
            n_id (int): Node whose class-probability we are interested in.
            n_cls (int): The class whose probability we ware interested in.
            X (np.ndarray): Feature matrix of graph including n_id.
            A (np.ndarray): Adjacency matrix of graph including n_id.
            y (np.ndarray): Node labels, entry for n_id ignored.

        Returns:
            float: Likelihood of node n_id being class n_cls in the given graph.
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

    def loglikelihood(self, n_id: int, n_cls: int, X: np.ndarray, A: np.ndarray, 
                   y: np.ndarray) -> float:
        """Calculate log-likelihood of a given node being of a certain class.
        
        Log-likelihood calculation is based on a given realized graph structure
        with the assumption the graph was generated using this CSBM model. 

        Args:
            n_id (int): Node whose class-probability we are interested in.
            n_cls (int): The class whose probability we are interested in.
            X (np.ndarray): Feature matrix of graph including n_id.
            A (np.ndarray): Adjacency matrix of graph including n_id.
            y (np.ndarray): Node labels, entry for n_id ignored.

        Returns:
            float: Log-likelihood of node n_id being class n_cls in the given graph.
        """
        likelihood = math.log(multivariate_normal.pdf(X[n_id,:], 
                                                      mean=(2*n_cls - 1)*self.mu, 
                                                      cov=self.cov))
        likelihood += self.structure_loglikelihood(n_id, n_cls, A, y)
        
        return likelihood

    def structure_loglikelihood(self, n_id: int, n_cls: int, A: np.ndarray, 
                                y: np.ndarray) -> float:
        """Calculate structural log-likelihood of a given node's label.
        
        Structural log-likelihood p(y_n | A) refers to the likelihood of a 
        node's label given the adjacency matrix ignoring its attributes.

        Args:
            n_id (int): Node whose class-probability we are interested in.
            n_cls (int): The class whose probability we are interested in.
            A (np.ndarray): Adjacency matrix of graph including n_id.
            y (np.ndarray): Node labels, entry for n_id ignored.

        Returns:
            float: Structural log-likelihood of a given node's label.
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

    def feature_separability(self, X: np.ndarray, y: np.ndarray, 
                             ids: Optional[List]=None) -> Tuple[int, int]:
        """Check (bayes) feature separability of graph X, y given CSBM.

        Feature separability refers to the fact if the base classifier would
        classify a given node correctly, if it only has access to the nodes
        features and not its incident edges.
        
        Optionally only check for nodes in ids.

        Args:
            X (np.ndarray): Feature matrix of given graph.
            y (np.ndarray): Labels of given graph.
            ids (Optional[List], optional): Check feature separability only for
                nodes in list ids. Defaults to None.

        Returns:
            Tuple[int, int]: #separable nodes, #non-separable nodes
        """
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

    def structure_separability(self, A: np.ndarray, y: np.ndarray, 
                               ids: Optional[List]=None) -> Tuple[int, int]:
        """Check (bayes) structural separability of graph A, y given CSBM.

        Structural separability refers to the correct classifciation of a given
        node based on the class-likelihood given the adjacency matrix ignoring 
        its attributes, i.e. is y* = max_y p(y|A) the actual class of the node?
        
        Optionally only check for nodes in ids.

        Args:
            A (np.ndarray): Adjacency matrix of given graph.
            y (np.ndarray): Labels of given graph.
            ids (Optional[List], optional): Check structure separability only 
                for nodes in list ids. Defaults to None.

        Returns:
            Tuple[int, int]: #separable nodes, #non-separable nodes
        """
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

    def likelihood_separability(self, X: np.ndarray, A: np.ndarray, 
                                y: np.ndarray, ids: Optional[List]=None) \
                                    -> Tuple[int, int]:
        """Check (bayes) separability of graph X, A, y given CSBM model.
        
        Separability refers to the correct classifciation of a given node
        based on the true (bayes) likelihood given the data model and the given 
        graph, i.e. is y* = max_y p(y|A, X, y/n_id) the actual class of the 
        node? Assumes all other labels are correct.

        Args:
            X (np.ndarray): Feature matrix of given graph.
            A (np.ndarray): Adjacency matrix of given graph.
            y (np.ndarray): Labels of given graph.
            ids (Optional[List], optional): Check structure separability only 
                for nodes in list ids. Defaults to None.

        Returns:
            Tuple[int, int]: #separable nodes, #non-separable nodes
        """
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