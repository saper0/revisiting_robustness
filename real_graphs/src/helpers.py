from rgnn_at_scale.data import prep_graph

import logging

from typing import Any, Dict, Iterable, List, Union, Tuple, Optional

import numpy as np
import torch
from torch_sparse import SparseTensor

from rgnn_at_scale.helper import utils
from rgnn_at_scale.data import split

from torch_geometric.datasets.planetoid import Planetoid


def load_cora(device: Union[int, str, torch.device] = 0,
                make_undirected: bool = True,
                binary_attr: bool = False,
                feat_norm: bool = False,
                dataset_root: str = 'data'):
    '''Loads the Cora dataset from 
    https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.Planetoid
    '''
    cora = Planetoid(root = dataset_root, name='cora')
    attr = cora.data.x.to(device)
    labels = cora.data.y.to(device)
    edge_index = cora.data.edge_index.to(device)
    edge_weight = torch.ones(edge_index.shape[1]).to(device)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight, sparse_sizes=(edge_index.max()+1, edge_index.max()+1)).to(device)

    # TODO make undirected
    if binary_attr:
        attr[attr != 0] = 1
    elif feat_norm:
        attr = utils.row_norm(attr)
    return attr, adj, labels

def prepare_data(dataset, data_device, dense_split=False, n_per_class = 20, make_undirected=True, binary_attr= False, feat_norm=False, data_dir='data'):
    '''Loads datasets and split nodes'''
    # load graph
    if dataset == 'Cora':
        attr, adj, labels = load_cora(data_device, make_undirected, binary_attr, feat_norm, data_dir)
    else:
        graph = prep_graph(name=dataset, device=data_device, dataset_root=data_dir, return_original_split=dataset.startswith('ogbn'), feat_norm=feat_norm)
        attr, adj, labels = graph[:3]
    # split
    if dataset.startswith('ogbn'):
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']
    else:
        if dense_split:
            idx_test, idx_val, idx_train = split(labels.cpu().numpy(), n_per_class=n_per_class)# changed order -> n_per_class for test and val
        else:
            idx_train, idx_unlabeled, idx_val, idx_test = split_inductive_sparse(labels.cpu().numpy(), n_per_class=n_per_class)
    logging.info(f"Training set size: {len(idx_train)}")
    logging.info(f"Validation set size: {len(idx_val)}")
    logging.info(f"Test set size: {len(idx_test)}")
    return attr, adj, idx_train, idx_val, idx_test, labels

def split_data(graph, labels, n_per_class, dense_split):
    if len(graph) == 3 or graph[3] is None:  # TODO: This is weird
        if dense_split:
            idx_test, idx_val, idx_train = split(labels.cpu().numpy(), n_per_class=n_per_class)# changed order -> n_per_class for test and val
        else:
            idx_train, idx_unlabeled, idx_val, idx_test = split_inductive_sparse(labels.cpu().numpy(), n_per_class=n_per_class)
    else:
        idx_train, idx_val, idx_test = graph[3]['train'], graph[3]['valid'], graph[3]['test']
        # for ogbn-arxiv subsample test idx 
        idx_keep = []
        for c in labels.unique():
            idx = (labels[idx_test] == c).nonzero().flatten()
            stop = min(len(idx), n_per_class)
            idx_keep.append(idx[torch.randperm(len(idx))[:stop]])
        idx_test = torch.cat(idx_keep).cpu().numpy()


    #n_features = attr.shape[1]
    n_classes = int(labels[~labels.isnan()].max() + 1)

    logging.info(f"Training set size: {len(idx_train)}")
    logging.info(f"Validation set size: {len(idx_val)}")
    logging.info(f"Test set size: {len(idx_test)}")
    return idx_train, idx_val, idx_test



def split_inductive_sparse(labels, n_per_class=20, seed=None):
    """
    Randomly split the training data.

    Parameters
    ----------
    labels: array-like [num_nodes]
        The class labels
    n_per_class : int or list[int]
        Number of samples per class
    seed: int
        Seed

    Returns
    -------
    split_labeled: array-like [n_per_class * nc]
        The indices of the training nodes
    split_val: array-like [n_per_class * nc]
        The indices of the validation nodes
    split_test: array-like [n_per_class * nc]
        The indices of the test nodes
    split_unlabeled: array-like [num_nodes - 3*n_per_class * nc]
        The indices of the unlabeled nodes
    """
    if seed is not None:
        np.random.seed(seed)
    if type(n_per_class)==int:
        n_train = n_per_class
        n_val = n_per_class
        n_test = n_per_class
    else:
        n_train = n_per_class[0]
        n_val = n_per_class[1]
        n_test = n_per_class[2]
    nc = labels.max() + 1

    split_labeled, split_val, split_test = [], [], []
    for label in range(nc):
        perm = np.random.permutation((labels == label).nonzero()[0])
        split_labeled.append(perm[:n_train])
        split_val.append(perm[n_train: n_train+n_val])
        split_test.append(perm[n_train+n_val: n_train+n_val+n_test])

    split_labeled = np.random.permutation(np.concatenate(split_labeled))
    split_val = np.random.permutation(np.concatenate(split_val))
    split_test = np.random.permutation(np.concatenate(split_test))

    assert split_labeled.shape[0] == n_train * nc
    assert split_val.shape[0] == n_val * nc
    assert split_test.shape[0] == n_test * nc

    split_unlabeled = np.setdiff1d(np.arange(len(labels)), np.concatenate((split_labeled, split_val, split_test)))

    print(f'number of samples\n - labeled: {n_train * nc} \n - val: {n_val * nc} \n - test: {n_test * nc} \n - unlabeled: {split_unlabeled.shape[0]}')

    return split_labeled, split_unlabeled, split_val, split_test



def delete_idx_from_data(attr, adj, idx):
    '''sets attr and adj entries corresponding to nodes in idx to 0'''
    attr_copy = attr.clone() # make sure to not overwrite attributes
    attr_copy[idx] = 0

    row, col, edge_weight = adj.t().coo()

    mapping = torch.ones(adj.size(dim=0)).bool()
    mapping[idx]=False # True if node not in idx

    mask_col = mapping[col] # True if col not in idx
    mask_row = mapping[row] # True if row not in idx
    mask_row_col = torch.logical_and(mask_col, mask_row) # True if none of row and col in idx -> entries we want to keep

    row = row[mask_row_col]
    col = col[mask_row_col]
    edge_weight = edge_weight[mask_row_col]

    return attr_copy, SparseTensor(row=row, col=col, value=edge_weight, sparse_sizes=(adj.size(dim=0), adj.size(dim=1)))