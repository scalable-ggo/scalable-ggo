import cupy as cp

from sggo import gpu_utils
from sggo.cluster import Cluster
from sggo.types import NDArray

from .lj_base import LJ


class LJAuto(LJ):
    def energies(self, cluster: Cluster) -> NDArray:
        return self._energies_shared(cluster.positions)

    def energy_gradient(self, cluster: Cluster) -> NDArray:
        return self._energy_gradient_shared(cluster.positions)


class LJCPU(LJ):
    def energies(self, cluster: Cluster) -> NDArray:
        pos = cp.asnumpy(cluster.positions)
        res = self._energies_shared(pos)
        return gpu_utils.assimilar(res, cluster.positions)

    def energy_gradient(self, cluster: Cluster) -> NDArray:
        pos = cp.asnumpy(cluster.positions)
        res = self._energy_gradient_shared(pos)
        return gpu_utils.assimilar(res, cluster.positions)


class LJGPU(LJ):
    def energies(self, cluster: Cluster) -> NDArray:
        pos = cp.asarray(cluster.positions)
        res = self._energies_shared(pos)
        return gpu_utils.assimilar(res, cluster.positions)

    def energy_gradient(self, cluster: Cluster) -> NDArray:
        pos = cp.asarray(cluster.positions)
        res = self._energy_gradient_shared(pos)
        return gpu_utils.assimilar(res, cluster.positions)
