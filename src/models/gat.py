from typing import Tuple, Union

from torch import nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GATv2Conv
from torch_sparse import SparseTensor
from torchtyping import TensorType

from src.models.utils import process_input

class GAT(nn.Module):
    """Two layer GAT implementation following Veličković et al. "Graph 
    Attention Netowrks", ICLR 2018. 
    
    Basically a wrapper for torch_geometric's GATConv. Default values 
    represent original published parametrization for Cora. 

    Parameters
    ----------
    n_features : int
        Number of attributes for each node
    n_classes : int
        Number of classes for prediction
    activation : nn.Module, optional
        Arbitrary activation function for the hidden layer, by default ELU.
    n_heads : int, optional
        Number of heads of first attention layer. (Final attention layer has
        only one head following the original GAT paper.) Default: 8
    n_features_per_head : int, optional
        Number of output features for each head of the first attention layer.
        Default: 8
    negative_slope : float, optional
        Angle of LeakyReLU's negative slope as used in the attention mechanism.
        Default: 0.2
    bias: bool, optional
        If set to False, the gat layers will not learn an additive bias.
        Default: True
    dropout : float, optional
        Dropout rate for input to attention layers, by default 0.6
    dropout_neighourhood : float, optional
        Dropout probability of the normalized attention coefficients which 
        exposes each node to a stochastically sampled neighborhood during 
        training., by default 0.6
    add_self_loops: bool, optional
        If set to False, will not add self-loops to the input graph. 
        (Default: True)
    """
    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 activation: Union[str, nn.Module] = nn.ELU(),
                 n_heads: int = 8, 
                 n_features_per_head: int = 8,
                 negative_slope: float = 0.2,
                 bias: bool = True,
                 dropout: float = 0.6,
                 dropout_neighourhood: float = 0.6,
                 add_self_loops: bool = True, 
                 gat_v2: bool = False,
                 **kwargs):
        super().__init__()
        if not gat_v2:
            self.gat1 = GATConv(in_channels=n_features, 
                                out_channels=n_features_per_head,
                                heads=n_heads, 
                                concat=True, 
                                negative_slope=negative_slope,
                                dropout=dropout_neighourhood,
                                add_self_loops=add_self_loops,
                                bias=bias)
            self.gat2 = GATConv(in_channels=n_heads*n_features_per_head,
                                out_channels=n_classes,
                                heads=1,
                                concat=False,
                                negative_slope=negative_slope,
                                dropout=dropout_neighourhood,
                                add_self_loops=add_self_loops,
                                bias=bias)
        else:
            self.gat1 = GATv2Conv(in_channels=n_features, 
                                out_channels=n_features_per_head,
                                heads=n_heads, 
                                concat=True, 
                                negative_slope=negative_slope,
                                dropout=dropout_neighourhood,
                                add_self_loops=add_self_loops,
                                bias=bias)
            self.gat2 = GATv2Conv(in_channels=n_heads*n_features_per_head,
                                out_channels=n_classes,
                                heads=1,
                                concat=False,
                                negative_slope=negative_slope,
                                dropout=dropout_neighourhood,
                                add_self_loops=add_self_loops,
                                bias=bias)
        self.dropout = nn.Dropout(dropout, inplace=False)
        self.activation = activation
        self.K = 2

    def forward(self, data: Union[Data, TensorType["n_nodes", "n_features"]], 
                      adj: Union[SparseTensor, 
                                 Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
                                 TensorType["n_nodes", "n_nodes"]]) -> Tensor:
        # Extract feature matrix, edge indices & values from arguments
        x, edge_index, edge_weight = process_input(data, adj)
        x = self.dropout(x)
        x = self.activation(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = self.gat2(x, edge_index)

        return x

