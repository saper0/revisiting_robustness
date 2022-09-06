from typing import Tuple, Union

from torch import nn, Tensor
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torchtyping import TensorType

from src.models.utils import process_input

class SGConv(torch_geometric.nn.SGConv):
    
    def __init__(self, dropout=0.5, **kwargs):
        super(SGConv, self).__init__(**kwargs)
        self.normalize = True
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        cache = self._cached_x
        if cache is None:
            if self.normalize:
                if isinstance(edge_index, Tensor):
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                elif isinstance(edge_index, SparseTensor):
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)

            for k in range(self.K):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                                   size=None)
                if self.cached:
                    self._cached_x = x
        else:
            x = cache

        return self.lin(self.dropout(x))


class SGC(nn.Module):
    r"""The simple graph convolutional operator from the `"Simplifying Graph
    Convolutional Networks" <https://arxiv.org/abs/1902.07153>`_ paper. 

    Basically a wrapper around the pytorch-geometric implementation adapted
    with feature-dropout.

    Args:
        in_channels (int): Size of each input sample, or -1 to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        K (int, optional): Number of hops "K". (default: 2)
        dropout (float): Dropout before applying logistic regression part.
            Default: 0.5
        cached (bool, optional): If set to True, the layer will cache
            the computation of (D^-1/2 A D^-1/2)^K * X on
            first execution, and will use the cached version for further
            executions.
            This parameter should only be set to True in transductive
            learning scenarios. (default: False)
        add_self_loops (bool, optional): If set to False, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to False, the layer will not learn
            an additive bias. (default: True)
    """
    def __init__(self, n_features: int, n_classes: int, K: int = 1,
                 dropout=0.5, cached: bool = False, 
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        super().__init__()
        self.sgc = SGConv(in_channels=n_features,
                          out_channels=n_classes,
                          K=K,
                          dropout=dropout,
                          cached=cached,
                          add_self_loops=add_self_loops,
                          bias=bias)

    def forward(self, data: Union[Data, TensorType["n_nodes", "n_features"]], 
                      adj: Union[SparseTensor, 
                                 Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
                                 TensorType["n_nodes", "n_nodes"]]) -> Tensor:
        x, edge_index, edge_weight = process_input(data, adj)
        x = self.sgc(x, edge_index, edge_weight)
        return x