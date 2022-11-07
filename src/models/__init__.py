from typing import Any, Dict, Union

from src.models.appnp import APPNP
from src.models.gat import GAT
from src.models.gcn import DenseGCN
from src.models.lp import LP
from src.models.mlp import MLP
from src.models.graphsage import GraphSAGE
from src.models.sgc import SGC


MODEL_TYPE = Union[APPNP, DenseGCN, GAT, SGC, MLP, GraphSAGE, None]

        
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
    if hyperparams['model'] == "DenseGCN":
        return DenseGCN(**hyperparams)
    if hyperparams['model'] == "LinearGCN":
        return DenseGCN(activation="Identity", **hyperparams)
    if hyperparams['model'] == "GAT":
        return GAT(**hyperparams)
    if hyperparams['model'] == "GATv2":
        return GAT(gat_v2=True, **hyperparams)
    if hyperparams['model'] is None:
        return None
    if hyperparams['model'] == "SGC":
        return SGC(**hyperparams)
    if hyperparams['model'] == "GraphSAGE":
        return GraphSAGE(**hyperparams)
    if hyperparams['model'] == "MLP":
        return MLP(**hyperparams)
    raise ValueError("Specified model not found.")


__all__ = [APPNP,
           GAT,
           DenseGCN,
           SGC,
           LP,
           MLP,
           GraphSAGE,
           create_model,
           MODEL_TYPE]
