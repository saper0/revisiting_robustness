from typing import Any, Dict, Tuple

import numpy as np

def calc_balanced_sample(class_counts: np.ndarray, n_samples: int) -> np.ndarray:
    """Return how many samples should be sampled from each class. 
    
    For a given number of nodes one wants to draw, returns how many nodes should
    be drawn from each class, such that the draw is as much class-balanced as 
    possible and accounting for the possibility that certain class_counts could 
    be exceeded if one draws naivly n_samples / classes samples per class.

    Args:
        class_counts (np.ndarray, [classes,]): Number of nodes for each class
            in the graph.
        n_samples (int): Nodes to sample in total (over all classes).

    Returns:
        np.ndarray [classes,]: (Balanced) number of nodes to samples per class.
    """
    n_c = len(class_counts)
    n_per_class = np.zeros(n_c, dtype=int)
    n_draw = n_samples
    if n_draw == 0:
        return n_per_class
    assert sum(class_counts) >= n_draw
    while True: 
        for c in range(n_c):
            if class_counts[c] > n_per_class[c]:
                n_per_class[c] += 1
                n_draw -= 1
                if n_draw == 0:
                    return n_per_class


def split(
    labels: np.ndarray, data_params: Dict[Any, str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly split the data in training and validation. 
    
    Training nodes are sampled in a class balanced fashion as far as possible.
    All other nodes are considered validation.

    Args:
        labels (np.ndarray, [num_nodes]): The class labels.
        data_params (Dict[Any, str]): Dict specifying the split, holds keys:
            "classes": Cardinality of label set
            "n_per_class_trn": Number of nodes per class in training set
            "n": Number of nodes in total

    Returns:
        A tuple (split_trn, split_val) with:
            split_trn (np.array, [n_per_class_trn * classes]): The indices of 
                the training nodes.
            split_val (np.array, [n - n_per_class_trn * classes]): The indices 
                of the validation nodes.
    """
    n_c = data_params["classes"]
    n_per_class_trn = data_params["n_per_class_trn"]
    class_counts = np.unique(labels, return_counts=True)[1]
    n_per_class = calc_balanced_sample(class_counts, 
                                       n_per_class_trn*n_c)

    split_trn = []
    for c in range(n_c):
        perm = np.random.permutation((labels == c).nonzero()[0])
        split_trn.append(perm[:n_per_class_trn])
    split_trn = np.random.permutation(np.concatenate(split_trn))
    assert split_trn.shape[0] == sum(n_per_class)

    split_val = np.setdiff1d(np.arange(len(labels)), split_trn)

    return split_trn, split_val