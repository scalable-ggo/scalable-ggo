from sggo.local_opt import LocalOpt
from sggo.energy import Energy
from sggo.cluster import Cluster

import cupy as cp


class FIRE(LocalOpt):
    maxstep: float
    dt_start: float
    dtmax: float
    nmin: int
    finc: float
    fdec: float
    a_start: float
    fa: float

    def __init__(
        self,
        energy: Energy,
        *,
        maxstep: float = 0.2,
        dt_start: float = 0.1,
        dtmax: float = 1.0,
        nmin: int = 5,
        finc: float = 1.1,
        fdec: float = 0.5,
        a_start: float = 0.1,
        fa: float = 0.99,
    ) -> None:
        super(FIRE, self).__init__(energy)
        self.maxstep = maxstep
        self.dt_start = dt_start
        self.dtmax = dtmax
        self.nmin = nmin
        self.finc = finc
        self.fdec = fdec
        self.a_start = a_start
        self.fa = fa

    def _fire(
        self,
        cluster: Cluster,
        target_gradient: float,
        max_steps: int,
    ) -> Cluster:
        pos = cluster.positions
        xp = cp.get_array_module(pos)

        target_gradient = xp.float32(target_gradient)
        maxstep = xp.float32(self.maxstep)
        dtmax = xp.float32(self.dtmax)
        nmin = self.nmin
        finc = xp.float32(self.finc)
        fdec = xp.float32(self.fdec)
        fa = xp.float32(self.fa)
        a_start = xp.float32(self.a_start)
        dt_start = xp.float32(self.dt_start)

        a = a_start
        dt = dt_start
        velocity = xp.zeros_like(pos, dtype=xp.float32)
        step = 0

        for _ in range(max_steps):
            gradient = self.energy.energy_gradient(cluster)

            norm = xp.linalg.norm(gradient, axis=1).max()
            if norm < target_gradient:
                break

            delta_e = xp.vdot(gradient, velocity)

            if delta_e < xp.float32(0):
                velocity = (xp.float32(1) - a) * velocity - a * gradient / xp.linalg.norm(gradient) * xp.linalg.norm(velocity)

                if step > nmin:
                    dt *= finc
                    dt = dtmax if dt > dtmax else dt
                    a *= fa

                step += 1
            else:
                velocity[:, :] = xp.float32(0)
                a = a_start
                dt *= fdec
                step = 0

            velocity += -gradient * dt
            delta_pos = velocity * dt

            step_len = xp.linalg.norm(delta_pos)

            if step_len > maxstep:
                delta_pos = delta_pos / step_len * maxstep

            pos += delta_pos

        return cluster
