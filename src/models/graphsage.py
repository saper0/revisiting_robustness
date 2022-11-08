from typing import Optional, Callable, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import ReLU
from torch_geometric.data import Data
from torch_geometric.nn.conv import SAGEConv
from torch_sparse import SparseTensor
from torchtyping import TensorType

from src.models.utils import process_input


class GraphSAGE(nn.Module):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing. 
    Implementation uses mean-aggregation.

    Args:
        n_features (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        n_classes (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        activation (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    def __init__(self, n_features: int, hidden_channels: int, 
                 n_classes: Optional[int] = None, dropout: float = 0.0,
                 activation: Optional[Callable] = ReLU(inplace=True),
                 **kwargs):
        super().__init__()

        self.sage1 = SAGEConv(n_features, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, n_classes)
        self.K = 2

        self.dropout = nn.Dropout(dropout, inplace=False)
        self.activation = activation

    def forward(self, data: Union[Data, TensorType["n_nodes", "n_features"]], 
                      adj: Union[SparseTensor, 
                                 Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
                                 TensorType["n_nodes", "n_nodes"]],
                      *args, **kwargs) -> Tensor:
        # Extract feature matrix, edge indices & values from arguments
        x, edge_index, edge_weight = process_input(data, adj)
        x = self.dropout(x)
        x = self.activation(self.sage1(x, edge_index))
        x = self.dropout(x)
        x = self.sage2(x, edge_index)

        return x