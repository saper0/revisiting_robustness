from typing import Any, Dict

from src.graph_models.csbm import CSBM

GRAPH_MODEL_TYPE = CSBM

def create_graph_model(hyperparams: Dict[str, Any]) -> GRAPH_MODEL_TYPE:
    """Initialize and return a graph model for synthetic graph generation.

    Args:
        hyperparams (Dict[str, Any]): Parameters for initializing graph model.

    Raises:
        ValueError: If a not implemented graph model is requested.

    Returns:
        GRAPH_MODEL_TYPE: Initialized graph model.
    """
    if hyperparams["graph_model"] == "CSBM":
        return CSBM(**hyperparams)
    raise ValueError("Specified graph model not found.")

__all__ = [CSBM, GRAPH_MODEL_TYPE, create_graph_model]