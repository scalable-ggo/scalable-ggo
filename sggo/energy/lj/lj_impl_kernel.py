import importlib.resources as rscs
import math
from typing import Any, Final

import cupy as cp

from sggo import gpu_utils
from sggo.cluster import Cluster
from sggo.types import NDArray

from .lj_base import LJ


class LJGPUKernel(LJ):
    pairwise_energy_kernel: Final[Any] = cp.RawKernel(rscs.read_text("sggo.energy.lj", "pairwise_energy.cu"), "pairwise_energy", backend="nvcc")

    pairwise_force_kernel: Final[Any] = cp.RawKernel(rscs.read_text("sggo.energy.lj", "pairwise_force.cu"), "pairwise_force", backend="nvcc")

    def energies(self, cluster: Cluster) -> NDArray:
        pos = cp.asarray(cluster.positions, dtype=cp.float32)

        n = len(pos)
        pairwise_energy = cp.zeros((n), dtype=cp.float32)

        warplines = math.ceil(n / 32)
        self.pairwise_energy_kernel((n,), (warplines * 32,), (n, pos, pairwise_energy), shared_mem=4 * warplines)

        return gpu_utils.assimilar(pairwise_energy, cluster.positions)

    def energy_gradient(self, cluster: Cluster) -> NDArray:
        pos = cp.asarray(cluster.positions, dtype=cp.float32)

        n = len(pos)
        pairwise_force = cp.zeros((n, 3), dtype=cp.float32)

        warplines = math.ceil(n / 32)
        self.pairwise_force_kernel((n,), (warplines * 32,), (n, pos, pairwise_force), shared_mem=4 * warplines * 3)

        return gpu_utils.assimilar(pairwise_force, cluster.positions)
