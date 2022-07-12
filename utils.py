import math

import numpy as np

from graph_models import CSBM

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


# This function creates the "Pi" vector for the model (the
# ${n_classes}-simplex vector giving relative community sizes) from
# the `community_size_slope`. Adapted from https://github.com/google-research/graphworld
def make_pi(n_classes: int, community_size_slope: float) -> np.ndarray:
    pi = np.array(range(n_classes)) * community_size_slope
    pi += np.ones(n_classes)
    pi /= np.sum(pi)
    return pi

# This function creates the "PropMat" matrix for the model (the square matrix
# giving inter-community Poisson means) from the config parameters, particularly
# `p_to_q_ratio`. Adapted from https://github.com/google-research/graphworld
def make_prop_mat(n_classes: int, p_to_q_ratio: float) -> np.ndarray:
    prop_mat = np.ones((n_classes, n_classes))
    np.fill_diagonal(prop_mat, p_to_q_ratio)
    return prop_mat

def compute_expected_edge_proportions(pi, prop_mat):
  """Computes expected edge proportions within and between communities.

  To get the exptected edge counts withing and between communities, multiply 
  the exptected edge proportions with the expected. 

  Adapted from https://github.com/google-research/graphworld

  Args:
    pi: interable of non-zero community size proportions. Must sum to 1.0, but
      this check is left to the caller of this internal function.
    prop_mat: square, symmetric matrix of community edge count rates. Entries
      must be non-negative, but this check is left to the caller.
  Returns:
    symmetric matrix with shape prop_mat.shape giving expected edge proportions.
  """
  scale = np.matmul(pi, np.matmul(prop_mat, pi)) 
  prob_mat = prop_mat / scale
  return np.outer(pi, pi) * prob_mat 

def compute_expected_edge_counts(n, avg_deg, pi, prop_mat):
  """Computes expected edge counts within and between communities.
  
  Adapted from https://github.com/google-research/graphworld
  
  Args:
    n: number of nodes in the graph
    avg_deg: average (expected) degree of each node
    pi: interable of non-zero community size proportions. Must sum to 1.0, but
      this check is left to the caller of this internal function.
    prop_mat: square, symmetric matrix of community edge count rates. Entries
      must be non-negative, but this check is left to the caller.
  """
  expected_edge_count = n * avg_deg
  W = compute_expected_edge_proportions(pi, prop_mat)
  return expected_edge_count * W