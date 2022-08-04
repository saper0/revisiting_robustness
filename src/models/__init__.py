from typing import Any, Dict, Union

from src.models.appnp import APPNP
from src.models.gat import GAT
from src.models.gcn import GCN, DenseGCN
from src.models.sgc import SGC


MODEL_TYPE = Union[APPNP, DenseGCN, GAT, GCN, SGC]

        
def create_model(hyperparams: Dict[str, Any]) -> MODEL_TYPE:
    """Creates the model instance given the hyperparameters.

    Args:
        hyperparams (Dict[str, Any]): Containing the hyperparameters.

    Raises:
        ValueError: If a not implemented model is requested.

    Returns:
        MODEL_TYPE: The created model instance.
    """
    if hyperparams['model'] == "APPNP":
        return APPNP(**hyperparams)
    if hyperparams['model'] == 'DenseGCN':
        return DenseGCN(**hyperparams)
    if hyperparams['model'] == "GAT":
        return GAT(**hyperparams)
    if hyperparams['model'] == 'GCN':
        return GCN(**hyperparams)
    if hyperparams['model'] == "SGC":
        return SGC(**hyperparams)
    raise ValueError("Specified model not found.")


__all__ = [GCN,
           APPNP,
           GAT,
           DenseGCN,
           create_model,
           MODEL_TYPE]
