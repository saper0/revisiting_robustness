# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Callable, List, Tuple, Union

from torch_geometric.typing import Adj

import copy

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.data import Data
from torch_geometric.nn.conv import GCNConv, SAGEConv, GINConv, GATConv, \
    SGConv, GATv2Conv, ARMAConv, FiLMConv, SuperGATConv, TransformerConv
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_sparse import SparseTensor
from torchtyping import TensorType

from src.models.utils import process_input


class BasicGNN(torch.nn.Module):
    r"""An abstract class for implementing basic GNN models.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last'):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act

        self.convs = ModuleList()

        self.norms = None
        if norm is not None:
            self.norms = ModuleList(
                [copy.deepcopy(norm) for _ in range(num_layers)])

        if jk != 'last':
            self.jk = JumpingKnowledge(jk, hidden_channels, num_layers)

        if out_channels is not None:
            self.out_channels = out_channels
            if jk == 'cat':
                self.lin = Linear(num_layers * hidden_channels, out_channels)
            else:
                self.lin = Linear(hidden_channels, out_channels)
        else:
            if jk == 'cat':
                self.out_channels = num_layers * hidden_channels
            else:
                self.out_channels = hidden_channels

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()
        if hasattr(self, 'jk'):
            self.jk.reset_parameters()
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self, data: Union[Data, TensorType["n_nodes", "n_features"]], 
                      adj: Union[SparseTensor, 
                                 Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
                                 TensorType["n_nodes", "n_nodes"]],
                      *args, **kwargs) -> Tensor:
        # Extract feature matrix, edge indices & values from arguments
        x, edge_index, edge_weight = process_input(data, adj)
        xs: List[Tensor] = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.norms is not None:
                x = self.norms[i](x)
            if self.act is not None:
                x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if hasattr(self, 'jk'):
                xs.append(x)

        x = self.jk(xs) if hasattr(self, 'jk') else x
        x = self.lin(x) if hasattr(self, 'lin') else x
        return x




class GCN(BasicGNN):
    r"""The Graph Neural Network from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper, using the
    :class:`~torch_geometric.nn.conv.GCNConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GCNConv`.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(GCNConv(in_channels, hidden_channels, **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, **kwargs))


class GraphSAGE(BasicGNN):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        n_features (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        n_classes (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """
    def __init__(self, n_features: int, hidden_channels: int, num_layers: int,
                 n_classes: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last', 
                 **kwargs):
        super().__init__(n_features, hidden_channels, num_layers,
                         n_classes, dropout, act, norm, jk)

        self.convs.append(SAGEConv(n_features, hidden_channels))
        for _ in range(1, num_layers):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels))


class GIN(BasicGNN):
    r"""The Graph Neural Network from the `"How Powerful are Graph Neural
    Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, using the
    :class:`~torch_geometric.nn.GINConv` operator for message passing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"last"`).
            (default: :obj:`"last"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GINConv`.
    """
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 out_channels: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True),
                 norm: Optional[torch.nn.Module] = None, jk: str = 'last',
                 **kwargs):
        super().__init__(in_channels, hidden_channels, num_layers,
                         out_channels, dropout, act, norm, jk)

        self.convs.append(
            GINConv(GIN.MLP(in_channels, hidden_channels), **kwargs))
        for _ in range(1, num_layers):
            self.convs.append(
                GINConv(GIN.MLP(hidden_channels, hidden_channels), **kwargs))

    @staticmethod
    def MLP(in_channels: int, out_channels: int) -> torch.nn.Module:
        return Sequential(
            Linear(in_channels, out_channels),
            BatchNorm1d(out_channels),
            ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )


class MLP(torch.nn.Module):
    r"""Multi-layer Perceptron.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (Callable, optional): The non-linear activation function to use.
            (default: :meth:`torch.nn.ReLU(inplace=True)`)
    """
    def __init__(self, n_features: int, hidden_channels: int, num_layers: int,
                 n_classes: Optional[int] = None, dropout: float = 0.0,
                 act: Optional[Callable] = ReLU(inplace=True), **kwargs):
        super().__init__()
        self.in_channels = n_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = act

        if n_classes is not None:
            self.out_channels = n_classes
        else:
            self.out_channels = hidden_channels

        layers = []
        layers.append(Linear(n_features, hidden_channels))
        layers.append(self.act)
        layers.append(torch.nn.Dropout(dropout))
        for i in range(1, num_layers):
            layers.append(Linear(hidden_channels, hidden_channels))
            layers.append(self.act)
            layers.append(torch.nn.Dropout(dropout))

        if n_classes is not None:
            layers.append(Linear(hidden_channels, n_classes))

        self.model = Sequential(*layers)

    def reset_parameters(self):
        for module in self.model:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, data: Union[Data, TensorType["n_nodes", "n_features"]], 
                      adj: Union[SparseTensor, 
                                 Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
                                 TensorType["n_nodes", "n_nodes"]]) -> Tensor:
        # Extract feature matrix, edge indices & values from arguments
        x, edge_index, edge_weight = process_input(data, adj)
        x = self.model(x)
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')

