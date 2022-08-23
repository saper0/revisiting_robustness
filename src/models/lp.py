# Label Propagation implementation with Code & Comments mainly taken from 
# PyTorch Geometric implementation of the Correct and Smooth Framework:
# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/correct_and_smooth.html #noqa
from typing import Optional, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.models import LabelPropagation
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked


patch_typeguard()


class LP(torch.nn.Module):
    r"""Label Propagation Module. 
    
    Implements Label Spreading as defined in Zhou et al. "Learning with Local 
    and Global Consistency" and used e.g. in the Correct & Smooth Framework. 

    Basically, this is a wrapper around the LabelPropagation implementation
    in PyTorch Geometric. Formally:

    .. math::
        \mathbf{\hat{z}}^{(0)}_i &= \begin{cases}
            \mathbf{y}_i, & \text{if }i\text{ is training node,}\\
            \mathbf{\hat{z}}_i, & \text{else}
        \end{cases}

    .. math::
        \mathbf{\hat{Z}}^{(\ell)} = \alpha \mathbf{D}^{-1/2}\mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{\hat{Z}}^{(\ell - 1)} +
        (1 - \alpha) \mathbf{\hat{Z}}^{(\ell - 1)}

    to obtain the final prediction :math:`\mathbf{\hat{Z}}^{(L_2)}`.

    Args:
        num_layers (int): The number of propagations :math:`L_2`.
        alpha (float): The :math:`\alpha` coefficient.
        num_classes (int): The number of different classes.
    """
    def __init__(self, num_layers: int, alpha: float, num_classes: int):
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        self.prop = LabelPropagation(num_layers, alpha)
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_classes = num_classes

    @typechecked
    def smooth(self, 
               y_soft: Union[TensorType["n", "d"], None], 
               y_true: TensorType["l"], 
               mask: Union[TensorType["l", int], TensorType["n", bool], 
                           np.ndarray, List[int]],
               A: TensorType["n", "n"], 
               normalize: bool = True) -> Tensor:
        """Perform label propagation on a given (partly) labeled graph.

        Args:
            y_soft (TensorType["n", "d"] | None): (unnormalized) Predictions of 
                node labels, e.g. logits of a GNN. If None assumes no 
                predictions available and generates a n x d tensor filled with
                zeros.
            y_true (TensorType["l"]): True labels of the labeled nodes.
            mask (TensorType["l", int] | TensorType["n", bool] | np.ndarray, 
                  List[int]): 
                A mask or index tensor denoting which nodes were used for 
                training. Type requirement is only to be able to index the 
                y_soft tensor.
            A: (TensorType["n", "n"]): (Dense) Adjacency Matrix.
            normalize (bool, optional): If y_soft should be normalized. 
                Defaults to True.

        Returns:
            Tensor: Resulting (soft) labeling from label propagation.
        """
        if y_soft is None:
            n = A.size(0)
            y_soft = torch.zeros((n,self.num_classes), device=y_true.device)
        if normalize:
            y_soft = self.softmax(y_soft)

        if y_true.dtype == torch.long and y_true.size(0) == y_true.numel():
            y_true = F.one_hot(y_true.view(-1), num_classes=self.num_classes)
            y_true = y_true.to(y_soft.dtype)
        y_soft[mask] = y_true

        edge_index = A.nonzero().t()
        return self.prop(y_soft, edge_index, edge_weight=None)

    def __repr__(self):
        return (f'{self.__class__.__name__}(\n'
                f'    num_layers={self.num_layers}, alpha={self.alpha}\n'
                ')')