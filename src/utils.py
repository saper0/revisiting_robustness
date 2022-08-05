from typing import Optional, Union

import numpy as np
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

@typechecked
def accuracy(logits: TensorType["n", "c"], labels: TensorType["n"], 
             split_idx: Optional[Union[np.ndarray, int]] = None) -> float:
    """Returns the accuracy for a tensor of logits and a list of lables.
    
    Optionally, split indices can be given. Then, only the nodes in the split
    will be used for the accuracy calculation.
    
    Args:
        logits (TensorType["n", "c"]): logits (`.argmax(1)` should return most 
            probable class).
        labels (TensorType["n"]): target labels
        split_idx (np.ndarray|int, optional): index or array with indices for 
            which accuracy should be evaluated. Defaults to None.

    Returns:
        float: Accuracy of logits w.r.t. given labels.
    """
    if split_idx is not None:
        return (logits.argmax(1)[split_idx] == labels[split_idx]).float().mean().item()
    else:
        return (logits.argmax(1) == labels).float().mean().item()