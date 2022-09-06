from typing import Tuple, Union

import torch
from torch_sparse import SparseTensor
from torchtyping import TensorType
from torch_geometric.data import Data

def process_input(data: Union[Data, TensorType["n_nodes", "n_features"]], 
                  adj: Union[SparseTensor, 
                             Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
                             TensorType["n_nodes", "n_nodes"]]) \
                        -> Union[TensorType["n_nodes", "n_features"], 
                                 TensorType[2, "nnz"],
                                 TensorType["nnz"]]:
    """Process (divers) input to be forwarded through model into standard form.
    
    Standard form is defined as pytorch_geometric modules expect input.

    Returns:
        x ... feature matrix (|V|, |F|)
        edge_index ... edge indices (2, |E|)
        edge_weights ... edge weights, tensor with |E| elements.
    """
    edge_weight = None
    if isinstance(data, Data):
        # PyTorch Geometric support
        x, edge_index = data.x, data.edge_index
    else:
        x = data
        if isinstance(adj, tuple):
            edge_index, edge_weight = adj[0], adj[1]
        elif isinstance(adj, SparseTensor):
            edge_idx_rows, edge_idx_cols, edge_weight = adj.coo()
            edge_index = torch.stack([edge_idx_rows, edge_idx_cols], dim=0)
        else:
            if not adj.is_sparse:
                adj = adj.to_sparse()
            edge_index, edge_weight = adj._indices(), adj._values()

    if edge_weight is None:
        edge_weight = torch.ones_like(edge_index[0], dtype=torch.float32)

    if edge_weight.dtype != torch.float32:
        edge_weight = edge_weight.float()

    return x, edge_index, edge_weight