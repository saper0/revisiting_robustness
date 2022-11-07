import collections
from typing import Tuple, Union
from torchtyping import patch_typeguard
from typeguard import typechecked

import torch

from torch import nn

from torch_sparse import SparseTensor

patch_typeguard()


class DenseGraphConvolution(nn.Module):
    """Dense GCN convolution layer for the FGSM attack that requires a gradient 
    towards the adjacency matrix.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        Parameters
        ----------
        in_channels : int
            Number of channels of the input
        out_channels : int
            Desired number of channels for the output (for trainable linear transform)
        """
        super().__init__()
        self._linear = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, arguments: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Prediction based on input.

        Parameters
        ----------
        arguments : Tuple[torch.Tensor, torch.Tensor]
            Tuple with two elements of the attributes and dense adjacency matrix

        Returns
        -------
        torch.Tensor
            The new embeddings
        """
        x, adj_matrix = arguments

        x_trans = self._linear(x)
        return adj_matrix @ x_trans


@typechecked
class DenseGCN(nn.Module):
    """Dense two layer GCN for the FGSM attack that requires a gradient towards 
    the adjacency matrix.
    """

    def __init__(self,
                 n_features: int,
                 n_classes: int,
                 n_filters: int = 64,
                 activation: str ="ReLU",
                 dropout: float = 0.5,
                 ** kwargs):
        """
        Parameters
        ----------
        n_features : int
            Number of attributes for each node
        n_classes : int
            Number of classes for prediction
        n_filters : int, optional
            number of dimensions for the hidden units, by default 80
        activation : str, optional
            Either ReLU or Identity (=> LinearGCN/SGC), by default 
            ReLU.
        dropout : int, optional
            Dropout rate, by default 0.5
        """
        super().__init__()
        self.n_features = n_features
        self.n_filters = n_filters
        self.n_classes = n_classes
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "Identity":
            self.activation = nn.Identity()
        else:
            raise ValueError("Only ReLU and Identity supportes as activation "
                            "functions.")
        self.dropout = dropout
        self.layers = nn.ModuleList([
            nn.Sequential(collections.OrderedDict([
                ('gcn_0', DenseGraphConvolution(in_channels=n_features,
                                                out_channels=n_filters)),
                ('activation_0', self.activation),
                ('dropout_0', nn.Dropout(p=dropout))
            ])),
            nn.Sequential(collections.OrderedDict([
                ('gcn_1', DenseGraphConvolution(in_channels=n_filters,
                                                out_channels=n_classes)),
                ('softmax_1', nn.LogSoftmax(dim=1))
            ]))
        ])

    @ staticmethod
    def normalize_dense_adjacency_matrix(adj: torch.Tensor) -> torch.Tensor:
        """Normalizes the adjacency matrix as proposed for a GCN by Kipf et al. 
        Moreover, it only uses the upper triangular matrix of the input to 
        obtain the right gradient towards the undirected adjacency matrix.

        Parameters
        ----------
        adj: torch.Tensor
            The weighted undirected [n x n] adjacency matrix.

        Returns
        -------
        torch.Tensor
            Normalized [n x n] adjacency matrix.
        """
        adj_norm = torch.triu(adj, diagonal=1) + torch.triu(adj, diagonal=1).T
        adj_norm.data[torch.arange(adj.shape[0]), torch.arange(adj.shape[0])] = 1
        deg = torch.diag(torch.pow(adj_norm.sum(axis=1), - 1 / 2))
        adj_norm = deg @ adj_norm @ deg
        return adj_norm

    def forward(self, x: torch.Tensor, adjacency_matrix: Union[torch.Tensor, 
                                                SparseTensor]) -> torch.Tensor:
        """Prediction based on input.

        Parameters
        ----------
        x : torch.Tensor
            Dense [n, d] tensor holding the attributes
        adjacency_matrix : torch.Tensor
            Dense [n, n] tensor for the adjacency matrix

        Returns
        -------
        torch.Tensor
            The predictions (after applying the softmax)
        """
        if isinstance(adjacency_matrix, SparseTensor):
            adjacency_matrix = adjacency_matrix.to_dense()
        adjacency_matrix = \
                DenseGCN.normalize_dense_adjacency_matrix(adjacency_matrix)
        for layer in self.layers:
            x = layer((x, adjacency_matrix))
        return x

