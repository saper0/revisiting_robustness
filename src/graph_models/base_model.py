from abc import ABC, abstractmethod
import logging
from typing import Tuple

import numpy as np

class GraphGenerationModel(ABC):
    """Base class for graph generation models."""

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
        logging.info(f"Feature Separability:")
        n_corr, n_wrong = self.feature_separability(X, y, ids)
        logging.info(f"n_corr: {n_corr}")
        logging.info(f"n_wrong: {n_wrong}")
        logging.info(f"Structure Separability:")
        n_corr, n_wrong = self.structure_separability(A, y, ids)
        logging.info(f"n_corr: {n_corr}")
        logging.info(f"n_wrong: {n_wrong}")
        logging.info(f"Likelihood Separability:")
        n_corr, n_wrong = self.likelihood_separability(X, A, y, ids)
        logging.info(f"n_corr: {n_corr}")
        logging.info(f"n_wrong: {n_wrong}")