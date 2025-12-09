import cupy as cp

from sggo.cluster import Cluster
from sggo.energy import Energy
from sggo.local_opt import LocalOpt
from sggo.local_opt.linesearch import linesearch_wolfe


class BFGS(LocalOpt):
    c1: float
    c2: float

    def __init__(self,
                 energy: Energy,
                 *,
                 c1: float = 1e-4,
                 c2: float = 0.9,
                 ) -> None:
        super(BFGS, self).__init__(energy)
        self.c1 = c1
        self.c2 = c2

    def _bfgs(
        self,
        cluster: Cluster,
        target_gradient: float,
        max_steps: int,
    ) -> Cluster:
        xp = cp.get_array_module(cluster.positions)
        target_gradient = xp.float32(target_gradient)

        idmat = xp.eye(cluster.positions.size, dtype=xp.float32)
        invH = idmat

        for _ in range(max_steps):
            gradient = self.energy.energy_gradient(cluster)
            norm = xp.linalg.norm(gradient, axis=1).max()
            if norm < target_gradient:
                break

            direction = (-invH @ gradient.flatten()).reshape(-1, 3)
            alpha = linesearch_wolfe(cluster, self.energy, direction, self.c1, self.c2)
            delta_pos = alpha * direction
            cluster.positions += delta_pos
            delta_pos = delta_pos.flatten()
            delta_gradient = (self.energy.energy_gradient(cluster) - gradient).flatten()

            pk = 1 / xp.dot(delta_pos, delta_gradient)
            mat1 = idmat - pk * delta_pos[:, xp.newaxis] * delta_gradient[xp.newaxis, :]
            mat2 = idmat - pk * delta_gradient[:, xp.newaxis] * delta_pos[xp.newaxis, :]
            invH = mat1 @ invH @ mat2 + pk * delta_pos[:, xp.newaxis] * delta_pos[xp.newaxis, :]

        return cluster
