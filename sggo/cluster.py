import numpy as np
from numpy.typing import ArrayLike
from typing import Callable


class Cluster:
    positions: ArrayLike

    def __init__(self, positions: ArrayLike) -> None:
        self.positions = positions

    def copy(self) -> "Cluster":
        return Cluster(self.positions)

    def deepcopy(self) -> "Cluster":
        return Cluster(self.positions.copy())

    @staticmethod
    def generate(num_atoms: int, energy_fn: Callable[["Cluster"], float], r: int) -> "Cluster":
        En = 1e16
        cluster = None

        while En > 1e15:
            pos = np.random.uniform(-1, 1, (num_atoms, 3)).astype(np.float32)
            pos /= np.sqrt((pos * pos).sum(1))[:, np.newaxis]
            pos *= np.cbrt(np.random.uniform(0, r, num_atoms))[:, np.newaxis]
            cluster = Cluster(pos)

            En = energy_fn(cluster)

        return cluster
