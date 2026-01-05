import importlib.resources as rscs
from typing import Any, Final

import cupy as cp

from sggo import gpu_utils
from sggo.cluster import Cluster

from .fire_base import FIRE


class FIREGPUKernel(FIRE):
    fire_kernel: Final[Any] = cp.RawKernel(rscs.read_text("sggo.local_opt.fire", "fire_lj.cu"), "fire_lj", backend="nvcc")

    def local_min(
        self,
        cluster: Cluster,
        *,
        target_gradient: float = 0.1,
        max_steps: int = 100000,
    ) -> Cluster:
        res = cluster.deepcopy()
        res.positions = cp.asarray(res.positions)

        n = len(res.positions)
        self.fire_kernel((1,), (n,), (res.positions,))

        res.positions = gpu_utils.assimilar(res.positions, cluster.positions)

        return res
