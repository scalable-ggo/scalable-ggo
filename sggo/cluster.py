from typing import Callable

import cupy as cp
import numpy as np

from sggo.types import NDArray


class Cluster:
    positions: NDArray

    def __init__(self, positions: NDArray) -> None:
        self.positions = positions

    def copy(self) -> "Cluster":
        return Cluster(self.positions)

    def deepcopy(self) -> "Cluster":
        return Cluster(self.positions.copy())

    @staticmethod
    def generate(num_atoms: int, energy_fn: Callable[["Cluster"], float], r: int) -> "Cluster":
        En = 1e16
        cluster = None

        while cluster is None or En > 1e15:
            pos = np.random.uniform(-1, 1, (num_atoms, 3)).astype(np.float32)
            pos /= np.sqrt((pos * pos).sum(1))[:, np.newaxis]
            pos *= np.cbrt(np.random.uniform(0, r, num_atoms))[:, np.newaxis]
            cluster = Cluster(pos)

            En = energy_fn(cluster)

        return cluster

    def save(self, name="cluster.atom") -> None:
        cp.asnumpy(self.positions).tofile(name)

    def load(self, name="cluster.atom") -> None:
        arr = np.fromfile(name, dtype=np.float32)

        assert arr.size % 3 != 0

        self.positions = arr.reshape(-1, 3)
