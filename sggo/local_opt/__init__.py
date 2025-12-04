from abc import ABC, abstractmethod

from sggo.cluster import Cluster
from sggo.energy import Energy


class LocalOpt(ABC):
    energy: Energy

    def __init__(self, energy: Energy) -> None:
        self.energy = energy

    @abstractmethod
    def local_min(
        self,
        cluster: Cluster,
        *,
        target_gradient: float = 0.1,
        max_steps: int = 100000,
    ) -> Cluster:
        raise NotImplementedError("Please Implement this method")
