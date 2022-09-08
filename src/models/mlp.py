from typing import Optional, Callable, Tuple, Union

import torch.nn as nn
from torch import Tensor
from torch.nn import ReLU
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torchtyping import TensorType

from src.models.utils import process_input


class MLP(nn.Module):
    r"""One-Layer MLP.

    Args:
        n_features (int): Size of each input sample.
        hidden_channels (int): Size of each hidden dimension.
        n_classes (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        activation (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
    """
    def __init__(self, n_features: int, hidden_channels: int,
                 n_classes: Optional[int] = None, dropout: float = 0.0,
                 activation: Optional[Callable] = ReLU(inplace=True),
                 **kwargs):
        super().__init__()

        self.lin1 = nn.Linear(n_features, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, n_classes)

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
        x = self.activation(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)

        return x