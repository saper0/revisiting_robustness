# File copied and modified from pytorch_geometric

from typing import Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ReLU
from torch_sparse import SparseTensor, matmul
from torchtyping import TensorType

from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor

from src.models.utils import process_input


class APPNP(MessagePassing):
    r"""The APPNP
    from the `"Predict then Propagate: Graph Neural Networks meet Personalized
    PageRank" <https://arxiv.org/abs/1810.05997>`_ paper

    .. math::
        \mathbf{X}^{(0)} &= MLP(\mathbf{X})

        \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
        \mathbf{X}^{(0)}

        \mathbf{X}^{\prime} &= \mathbf{X}^{(K)},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        K (int): Number of iterations :math:`K`.
        alpha (float): Teleport probability :math:`\alpha`.
        dropout (float, optional): Dropout probability of edges during
            training. (default: :obj:`0`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          adjacency matrix SparseTensor or Tensor
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, n_features: int, n_classes: int, n_hidden: int, K: int, 
                 alpha: float, dropout: float = 0., cached: bool = False, 
                 add_self_loops: bool = True, normalize: bool = True, **kwargs):
        super().__init__(aggr = "add")
        # MLP
        self.lin1 = Linear(n_features, n_hidden)
        self.lin2 = Linear(n_hidden, n_classes)
        self.relu = ReLU()

        # Personalized Pagerank
        self.K = K
        self.alpha = alpha
        self.dropout = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, data: Union[Data, TensorType["n_nodes", "n_features"]], 
                      adj: Union[SparseTensor, 
                                 Tuple[TensorType[2, "nnz"], TensorType["nnz"]],
                                 TensorType["n_nodes", "n_nodes"]]) -> Tensor:
        """"""
        # Extract feature matrix, edge indices & values from arguments
        x, edge_index, edge_weight = process_input(data, adj)

        # Normalize
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # MLP
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        h = x
        # Personalized Pagerank
        for k in range(self.K):
            if self.dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.dropout)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.dropout)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            x = x * (1 - self.alpha)
            x += self.alpha * h

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'
