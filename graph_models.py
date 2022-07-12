from abc import ABC, abstractmethod
import math
from typing import Tuple

import numpy as np
from scipy.stats import multivariate_normal


class DataGenModel(ABC):

    @abstractmethod
    def sample(self, n) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a sample graph with n-nodes."""
        pass
    
    @abstractmethod
    def sample_conditional(self, n, X: np.ndarray, A: np.ndarray, y: np.ndarray) \
                                 -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample X', A', y' ~ D_n(X, A, y)"""
        pass

    @abstractmethod
    def loglikelihood(self, n_id, n_cls, X, A, y):
        """Return loglikelihood of node n_id being class n_cls given graph X,A,y.
        
        Graph must be complete graph including n_id, likelihood is calculated
        as if the whole graph except label of n_id-th node is known.
        """
        pass

    @abstractmethod
    def structure_loglikelihood(self, n_id, n_cls, A, y):
        """Return structure log-likelihood of node n_id being of class n_cls.
        
        Graph X,A,y must be complete graph including n_id, likelihood is 
        calculated as if the whole graph except label of n_id-th node is known.
        """
        pass

    @abstractmethod
    def feature_separability(self, X, y, ids=None) -> Tuple[int, int]:
        """Check (bayes) feature separability of graph X, y given (data-gen) 
        model.
        
        Optionally only check for nodes in ids.
        
        Return tuple: #separable, #non-separable."""
        pass

    @abstractmethod
    def structure_separability(self, A, y, ids=None) -> Tuple[int, int]:
        """Return tuple: #separable, #non-separable."""
        pass

    @abstractmethod
    def likelihood_separability(self, X, A, y, ids=None) -> Tuple[int, int]:
        """Return tuple: #separable, #non-separable."""
        pass

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


class CSBM(DataGenModel):
    """X, A ~ CSBM(n, p, q, mu, cov)"""
    def __init__(self, p, q, mu, cov) -> None:
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


from utils import make_pi, make_prop_mat, compute_expected_edge_counts


class DCSBM(DataGenModel):
    """X, A ~ DC-SBM(n, p_q_ratio, avg_deg, alpha, mu, cov, n_classes, 
                     community_size_slope)
    
    alpha ... power law exponent
    """
    def __init__(self, n, p_q_ratio, avg_deg, alpha, mu, cov, n_classes, 
                 community_size_slope) -> None:
        self.pi = make_pi(n_classes, community_size_slope)
        self.prop_mat = make_prop_mat(n_classes, p_q_ratio)
        self.W = compute_expected_edge_counts(n, avg_deg, self.pi, 
                                              self.prop_mat)

    def sample(self, n) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a sample graph with n-nodes."""
        pass
    
    def sample_conditional(self, n, X: np.ndarray, A: np.ndarray, y: np.ndarray) \
                                 -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample X', A', y' ~ D_n(X, A, y)"""
        pass

    def loglikelihood(self, n_id, n_cls, X, A, y):
        """Return loglikelihood of node n_id being class n_cls given graph X,A,y.
        
        Graph must be complete graph including n_id, likelihood is calculated
        as if the whole graph except label of n_id-th node is known.
        """
        pass

    def structure_loglikelihood(self, n_id, n_cls, A, y):
        """Return structure log-likelihood of node n_id being of class n_cls.
        
        Graph X,A,y must be complete graph including n_id, likelihood is 
        calculated as if the whole graph except label of n_id-th node is known.
        """
        pass

    def feature_separability(self, X, y, ids=None) -> Tuple[int, int]:
        """Check (bayes) feature separability of graph X, y given (data-gen) 
        model.
        
        Optionally only check for nodes in ids.
        
        Return tuple: #separable, #non-separable."""
        pass

    def structure_separability(self, A, y, ids=None) -> Tuple[int, int]:
        """Return tuple: #separable, #non-separable."""
        pass

    def likelihood_separability(self, X, A, y, ids=None) -> Tuple[int, int]:
        """Return tuple: #separable, #non-separable."""
        pass


class BAC(DataGenModel):
    """X, A ~ BA-C(p, q, m, mu, cov, pi)

    Contextual Barabasi-Albert model with community structure.
    p ... inner-class affinity
    q ... inter-class affinity
    m ... number of edges per iteration
    mu ... class dependent mean, must be of size len(pi) x d, check left to caller
    """
    def __init__(self, p, q, m, mu, cov, pi) -> None:
        self.pi = pi 
        self.p = p
        self.q = q
        self.m = m
        self.mu = mu
        self.d = self.mu[0].shape[0]
        self.cov = cov

    def sample(self, n, seed=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample y~Bin(1/2) and X,A ~ CBA-C(n, p, q, m, mu, cov, pi). 
        
        Return X: np.ndarray, 
               A: np.ndarray, 
               y: np.ndarray."""
        np.random.seed(seed)

        # Prepare X, A, y
        y = np.zeros((n,)).astype(int)
        X = np.zeros((n,self.d))
        A = np.zeros((n,n))
        deg = np.ones(n,) # add each node with a virtual self-loop / initial attractiveness of 1
        affinity = np.zeros(n,)

        for i in range(n):
            # Draw Label
            y[i] = np.argmax(np.random.multinomial(1, self.pi))
            # Draw Features
            X[i,:] = np.random.multivariate_normal(self.mu[y[i]], self.cov, 1)
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
                A[:i,i] = draw
        A += A.T
        return X.astype(np.float32), A.astype(int), y
        
    
    def sample_conditional(self, n, X: np.ndarray, A: np.ndarray, y: np.ndarray) \
                                 -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample X', A', y' ~ D_n(X, A, y)"""
        pass

    def loglikelihood(self, n_id, n_cls, X, A, y):
        """Return loglikelihood of node n_id being class n_cls given graph X,A,y.
        
        Graph must be complete graph including n_id, likelihood is calculated
        as if the whole graph except label of n_id-th node is known.
        """
        pass

    def structure_loglikelihood(self, n_id, n_cls, A, y):
        """Return structure log-likelihood of node n_id being of class n_cls.
        
        Graph X,A,y must be complete graph including n_id, likelihood is 
        calculated as if the whole graph except label of n_id-th node is known.
        """
        pass

    def feature_separability(self, X, y, ids=None) -> Tuple[int, int]:
        """Check (bayes) feature separability of graph X, y given (data-gen) 
        model.
        
        Optionally only check for nodes in ids.
        
        Return tuple: #separable, #non-separable."""
        pass

    def structure_separability(self, A, y, ids=None) -> Tuple[int, int]:
        """Return tuple: #separable, #non-separable."""
        pass

    def likelihood_separability(self, X, A, y, ids=None) -> Tuple[int, int]:
        """Return tuple: #separable, #non-separable."""
        pass


class BACD(DataGenModel):
    """X, A ~ BA-CD(p, q, mu, cov, pi)

    Contextual Barabasi-Albert model with community structure and variable degrees
    p ... inner-class affinity
    q ... inter-class affinity
    m ... number of edges per iteration
    mu ... class dependent mean, must be of size len(pi) x d, check left to caller
    """
    def __init__(self, p, q, mu, cov, pi) -> None:
        self.pi = pi 
        self.p = p
        self.q = q
        self.mu = mu
        self.d = self.mu[0].shape[0]
        self.cov = cov

    def sample(self, n, seed=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample y~Bin(1/2) and X,A ~ CBA-C(n, p, q, m, mu, cov, pi). 
        
        Return X: np.ndarray, 
               A: np.ndarray, 
               y: np.ndarray."""
        np.random.seed(seed)
        print("entered sample")
        # Prepare X, A, y
        y = np.zeros((n,)).astype(int)
        X = np.zeros((n,self.d))
        A = np.zeros((n,n))
        deg = np.ones(n,) # add each node with a virtual self-loop
        for i in range(n):
            # Draw Label
            y[i] = np.argmax(np.random.multinomial(1, self.pi))
            # Draw Features
            X[i,:] = np.random.multivariate_normal(self.mu[y[i]], self.cov, 1)
            idx_same = y[:i] == y[i]
            n_same = np.sum(idx_same)
            idx_diff = y[:i] != y[i]
            n_diff = np.sum(idx_diff)
            if n_same > 0:
                # Form Intra-Edges using PA
                n_draw = np.random.binomial(n_same, self.p)
                p_draw = deg[:i][idx_same] / np.sum(deg[:i][idx_same])
                draw = np.random.multinomial(n_draw, p_draw) > 0
                idx_same = idx_same.nonzero()[0] # Get indices from boolean array
                idx_chosen = idx_same[draw]
                deg[:i][idx_chosen] += 1
                A[:i, i][idx_chosen] = 1
            if n_diff > 0:
                # Form Inter-Edges using PA
                n_draw = np.random.binomial(n_diff, self.q)
                p_draw = deg[:i][idx_diff] / np.sum(deg[:i][idx_diff])
                draw = np.random.multinomial(n_draw, p_draw) > 0
                idx_diff = idx_diff.nonzero()[0] # Get indices from boolean array
                idx_chosen = idx_diff[draw]
                deg[:i][idx_chosen] += 1
                A[:i, i][idx_chosen] = 1
        A += A.T
        return X.astype(np.float32), A.astype(int), y
        
    
    def sample_conditional(self, n, X: np.ndarray, A: np.ndarray, y: np.ndarray) \
                                 -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample X', A', y' ~ D_n(X, A, y)"""
        pass

    def loglikelihood(self, n_id, n_cls, X, A, y):
        """Return loglikelihood of node n_id being class n_cls given graph X,A,y.
        
        Graph must be complete graph including n_id, likelihood is calculated
        as if the whole graph except label of n_id-th node is known.
        """
        pass

    def structure_loglikelihood(self, n_id, n_cls, A, y):
        """Return structure log-likelihood of node n_id being of class n_cls.
        
        Graph X,A,y must be complete graph including n_id, likelihood is 
        calculated as if the whole graph except label of n_id-th node is known.
        """
        pass

    def feature_separability(self, X, y, ids=None) -> Tuple[int, int]:
        """Check (bayes) feature separability of graph X, y given (data-gen) 
        model.
        
        Optionally only check for nodes in ids.
        
        Return tuple: #separable, #non-separable."""
        pass

    def structure_separability(self, A, y, ids=None) -> Tuple[int, int]:
        """Return tuple: #separable, #non-separable."""
        pass

    def likelihood_separability(self, X, A, y, ids=None) -> Tuple[int, int]:
        """Return tuple: #separable, #non-separable."""
        pass
