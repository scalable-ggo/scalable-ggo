import importlib.resources as rscs

import cupy as cp

from sggo import gpu_utils
from sggo.cluster import Cluster

from .fire_base import FIRE


class FIREGPUKernel(FIRE):

    def local_min(
        self,
        cluster: Cluster,
        *,
        target_gradient: float = 0.1,
        max_steps: int = 100000,
    ) -> Cluster:
        fire_kernel = cp.RawKernel(
                self._header(target_gradient, max_steps) +
                rscs.read_text("sggo.local_opt.fire", "fire_lj.cu"),
                "fire_lj", backend="nvcc")
        res = cluster.deepcopy()
        res.positions = cp.asarray(res.positions)

        n = len(res.positions)
        fire_kernel((1,), (n,), (res.positions,))

        res.positions = gpu_utils.assimilar(res.positions, cluster.positions)

        return res

    def _header(self, target_gradient: float, max_steps: int):
        return (
            f"#define MAXSTEP {self.maxstep}f\n"
            f"#define DT_START {self.dt_start}f\n"
            f"#define DTMAX {self.dtmax}f\n"
            f"#define NMIN {self.nmin}\n"
            f"#define FINC {self.finc}f\n"
            f"#define FDEC {self.fdec}f\n"
            f"#define A_START {self.a_start}f\n"
            f"#define FA {self.fa}f\n"
            f"#define MAX_STEPS {max_steps}\n"
            f"#define TARGET_GRADIENT {target_gradient}f\n"
        )
