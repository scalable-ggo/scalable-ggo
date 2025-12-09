import cupy as cp

from sggo.cluster import Cluster
from sggo.energy import Energy
from sggo.local_opt import LocalOpt
from sggo.local_opt.linesearch import linesearch_wolfe


class CG(LocalOpt):
    c1: float
    c2: float

    def __init__(self,
                 energy: Energy,
                 *,
                 c1: float = 1e-4,
                 c2: float = 0.4,
                 ) -> None:
        super(CG, self).__init__(energy)
        self.c1 = c1
        self.c2 = c2

    def _cg(
        self,
        cluster: Cluster,
        target_gradient: float,
        max_steps: int,
    ) -> Cluster:
        xp = cp.get_array_module(cluster.positions)
        target_gradient = xp.float32(target_gradient)

        gradient = self.energy.energy_gradient(cluster)
        direction = -gradient

        for _ in range(max_steps):
            norm = xp.linalg.norm(gradient, axis=1).max()
            if norm < target_gradient:
                break

            alpha = linesearch_wolfe(cluster, self.energy, direction, self.c1, self.c2)
            cluster.positions += alpha * direction

            gradient_old, gradient = gradient, self.energy.energy_gradient(cluster)
            beta = xp.vdot(gradient, gradient - gradient_old) / xp.vdot(gradient_old, gradient_old)
            beta = max(beta, xp.float32(0.0))
            direction = -gradient + beta * direction

        return cluster
