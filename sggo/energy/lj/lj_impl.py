import cupy as cp
from numpy.typing import ArrayLike

from .lj_base import LJ
from sggo.cluster import Cluster
from sggo import gpu_utils


class LJAuto(LJ):
    def energies(self, cluster: Cluster) -> ArrayLike:
        return self._energies_shared(cluster.positions)

    def energy_gradient(self, cluster: Cluster) -> ArrayLike:
        return self._energy_gradient_shared(cluster.positions)


class LJCPU(LJ):
    def energies(self, cluster: Cluster) -> ArrayLike:
        pos = cp.asnumpy(cluster.positions)
        res = self._energies_shared(pos)
        return gpu_utils.assimilar(res, cluster.positions)

    def energy_gradient(self, cluster: Cluster) -> ArrayLike:
        pos = cp.asnumpy(cluster.positions)
        res = self._energy_gradient_shared(pos)
        return gpu_utils.assimilar(res, cluster.positions)


class LJGPU(LJ):
    def energies(self, cluster: Cluster) -> ArrayLike:
        pos = cp.asarray(cluster.positions)
        res = self._energies_shared(pos)
        return gpu_utils.assimilar(res, cluster.positions)

    def energy_gradient(self, cluster: Cluster) -> ArrayLike:
        pos = cp.asarray(cluster.positions)
        res = self._energy_gradient_shared(pos)
        return gpu_utils.assimilar(res, cluster.positions)
