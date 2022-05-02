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

    def sample(self, n, seed = 0) -> Tuple[TensorType["n", "d"], 
                                           TensorType["n", "n"],
                                           TensorType["n"]]:
        """Sample y~Bin(1/2) and X,A ~ CSBM(n, p, q, mu, cov). 
        
        Return X: torch.tensor, 
               A: torch.tensor, 
               y: torch.tensor."""
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

    def feature_separability(self, X, y):
        """Check (bayes) feature separability of graph X, A, y given CSBM."""
        print(f"Feature Separability:")
        n_corr = 0
        n_wrong = 0
        for i in range(len(y)):
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
        print(f"n_corr: {n_corr}")
        print(f"n_wrong: {n_wrong}")

    def structure_separability(self, A, y):
        # Check how much nodes are correct w.r.t. structure likelihood
        print(f"Structure Separability:")
        n_corr = 0
        n_wrong = 0
        for i in range(len(y)):
            likelihood_corr = self.structure_loglikelihood(i, y[i], A, y)
            likelihood_wrong = self.structure_loglikelihood(i, -(y[i]-1), A, y)
            if likelihood_corr > likelihood_wrong:
                n_corr += 1
            else:
                n_wrong += 1
        print(f"n_corr: {n_corr}")
        print(f"n_wrong: {n_wrong}")

    def likelihood_separability(self, X, A, y):
        # Check how much nodes are correct w.r.t. likelihood
        print(f"Likelihood Separability:")
        n_corr = 0
        n_wrong = 0
        for i in range(len(y)):
            likelihood_corr = self.loglikelihood(i, y[i], X, A, y)
            likelihood_wrong = self.loglikelihood(i, -(y[i]-1), X, A, y)
            if likelihood_corr > likelihood_wrong:
                n_corr += 1
            else:
                n_wrong += 1
        print(f"n_corr: {n_corr}")
        print(f"n_wrong: {n_wrong}")

    def check_separabilities(self, X, A, y):
        self.feature_separability(X, y)
        self.structure_separability(A, y)
        self.likelihood_separability(X, A, y)

       