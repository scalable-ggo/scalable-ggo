from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from sggo.cluster import Cluster

import cupy as cp


class Energy(ABC):
    @abstractmethod
    def energies(self, cluster: Cluster) -> ArrayLike:
        raise NotImplementedError("Please Implement this method")

    def energy(self, cluster: Cluster) -> float:
        xp = cp.get_array_module(cluster.positions)
        return xp.sum(self.energies(cluster))

    @abstractmethod
    def energy_gradient(self, cluster: Cluster) -> ArrayLike:
        raise NotImplementedError("Please Implement this method")
