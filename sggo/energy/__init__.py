from abc import ABC, abstractmethod

import cupy as cp

from sggo.cluster import Cluster
from sggo.types import NDArray


class Energy(ABC):
    @abstractmethod
    def energies(self, cluster: Cluster) -> NDArray:
        raise NotImplementedError("Please Implement this method")

    def energy(self, cluster: Cluster) -> float:
        xp = cp.get_array_module(cluster.positions)
        return xp.sum(self.energies(cluster))

    @abstractmethod
    def energy_gradient(self, cluster: Cluster) -> NDArray:
        raise NotImplementedError("Please Implement this method")
