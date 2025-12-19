import heapq
import random
from typing import Callable, Optional

import cupy as cp
import numpy as np

from sggo import gpu_utils
from sggo.types import NDArray


class Cluster:
    positions: NDArray

    def __init__(self, positions: NDArray) -> None:
        self.positions = positions

    def __eq__(self, other: "Cluster") -> bool:
        if not isinstance(other, self.__class__):
            return False
        if not isinstance(other.positions, self.positions.__class__):
            return False

        return (self.positions == other.positions).all()

    __hash__ = None

    def __repr__(self) -> str:
        return f"Cluster({self.positions})"

    def copy(self) -> "Cluster":
        return Cluster(self.positions)

    def deepcopy(self) -> "Cluster":
        return Cluster(self.positions.copy())

    def ensure_seperation(self, seperation: float = 0.1) -> None:
        positions = cp.asnumpy(self.positions)
        buckets = set()

        for i in range(len(positions)):
            pos = positions[i]
            bucket = np.floor(pos / seperation)

            heap = [(0, np.zeros(3))]
            cube_len = 1

            def push_cube(heap, cube_len):
                cube_dist = cube_len**2
                for j in range(-cube_len, cube_len + 1):
                    for k in range(-cube_len, cube_len + 1):
                        heapq.heappush(heap, (cube_dist, [-cube_len, j, k]))
                        heapq.heappush(heap, (cube_dist, [cube_len, j, k]))
                        if j in {-cube_len, cube_len}:
                            continue

                        heapq.heappush(heap, (cube_dist, [j, -cube_len, k]))
                        heapq.heappush(heap, (cube_dist, [j, cube_len, k]))
                        if k in {-cube_len, cube_len}:
                            continue

                        heapq.heappush(heap, (cube_dist, [j, k, -cube_len]))
                        heapq.heappush(heap, (cube_dist, [j, k, cube_len]))

            found = False
            while not found:
                if len(heap) == 0 or cube_len**2 <= heap[0][0]:
                    push_cube(heap, cube_len)
                    cube_len += 1

                dist = heap[0][0]
                shifts = []
                while len(heap) > 0 and heap[0][0] <= dist:
                    shifts.append(np.array(heapq.heappop(heap)[1]))
                random.shuffle(shifts)

                for shift in shifts:
                    if tuple(bucket + shift) not in buckets:
                        pos += seperation * shift
                        bucket += shift
                        found = True
                        break

            for dx in -1, 0, 1:
                for dy in -1, 0, 1:
                    for dz in -1, 0, 1:
                        buckets.add((bucket[0] + dx, bucket[1] + dy, bucket[2] + dz))

        self.positions = gpu_utils.assimilar(positions, self.positions)

    @staticmethod
    def generate(
        num_atoms: int, density: float = 1, seperation: float = 0.1, rng: Optional[Callable[[float, float, int], NDArray]] = None
    ) -> "Cluster":
        if rng is None:
            rng = lambda lo, hi, size: np.random.uniform(lo, hi, size).astype(np.float32)
        maxr3 = 3 * num_atoms / 4 / np.pi / density

        distance = np.cbrt(rng(0, maxr3, num_atoms))
        angle = rng(0, 2 * np.pi, num_atoms)
        zuniform = rng(-1, 1, num_atoms)

        rsurface = np.sqrt(1 - zuniform**2)
        z = zuniform * distance
        x = rsurface * np.cos(angle) * distance
        y = rsurface * np.sin(angle) * distance

        cluster = Cluster(np.stack((x, y, z), axis=1))
        cluster.ensure_seperation(seperation=seperation)

        return cluster

    def save(self, name: str = "cluster.atoms") -> None:
        np.savetxt(name, cp.asnumpy(self.positions), fmt="%.10f")

    @classmethod
    def load(cls, name: str = "cluster.atoms") -> "Cluster":
        positions = np.loadtxt(name, dtype=np.float32)
        return cls(positions)
