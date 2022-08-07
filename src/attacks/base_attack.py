from abc import ABC, abstractmethod
from typing import Tuple, Union

class LocalAttack(ABC):
    """Provides possibility to attack a target node through changing one edge."""

    @abstractmethod
    def create_adversarial_pert(self) -> Tuple[int, int]:
        """Add or remove one edge in the stored graph.

        Returns:
            Tuple[int, int]: Node-index-tuple of newly added or deleted edge or 
            None if no perturbation possible anymore.
        """
        pass