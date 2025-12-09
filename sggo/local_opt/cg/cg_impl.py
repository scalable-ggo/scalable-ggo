from .cg_base import CG
from sggo import gpu_utils
from sggo.cluster import Cluster

import cupy as cp


class CGAuto(CG):
    def local_min(
            self,
            cluster: Cluster,
            *,
            target_gradient: float = 0.1,
            max_steps: int = 100000,
            ) -> Cluster:
        return self._cg(cluster.deepcopy(), target_gradient, max_steps)


class CGCPU(CG):
    def local_min(
            self,
            cluster: Cluster,
            *,
            target_gradient: float = 0.1,
            max_steps: int = 100000,
            ) -> Cluster:
        res = cluster.deepcopy()
        res.positions = cp.asnumpy(res.positions)
        res = self._cg(res, target_gradient, max_steps)
        res.positions = gpu_utils.assimilar(res.positions, cluster.positions)
        return res


class CGGPU(CG):
    def local_min(
            self,
            cluster: Cluster,
            *,
            target_gradient: float = 0.1,
            max_steps: int = 100000,
            ) -> Cluster:
        res = cluster.deepcopy()
        res.positions = cp.asarray(res.positions)
        res = self._cg(res, target_gradient, max_steps)
        res.positions = gpu_utils.assimilar(res.positions, cluster.positions)
        return res
