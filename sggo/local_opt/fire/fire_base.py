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
    astart: float
    fa: float

    def __init__(self,
                 energy: Energy,
                 *,
                 maxstep: float = 0.2,
                 dt_start: float = 0.1,
                 dtmax: float = 1.0,
                 nmin: int = 5,
                 finc: float = 1.1,
                 fdec: float = 0.5,
                 a_start: float = 0.1,
                 fa: float = 0.1,
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

    # TODO either reference ASE or reimplement
    def _fire(
        self,
        cluster: Cluster,
        target_gradient: float,
        max_steps: int,
    ) -> Cluster:
        pos = cluster.positions
        xp = cp.get_array_module(pos)

        fmax = xp.float32(target_gradient)
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

        v = None
        step_cnt = 0

        for _ in range(max_steps):
            cluster.positions = pos
            f = -self.energy.energy_gradient(cluster)
            norm = xp.linalg.norm(f, axis=1).max()
            if norm < fmax:
                break

            if v is None:
                v = xp.zeros_like(pos, dtype=xp.float32)
            else:
                vf = xp.vdot(f, v)
                if vf > xp.float32(0.0):
                    v = (xp.float32(1.0) - a) * v + a * f / xp.sqrt(
                        xp.vdot(f, f)) * xp.sqrt(xp.vdot(v, v))
                    if step_cnt > nmin:
                        dt = min(dt * finc, dtmax)
                        a *= fa
                    step_cnt += 1
                else:
                    v *= xp.float32(0.0)
                    a = a_start
                    dt *= fdec
                    step_cnt = 0

            v += dt * f
            dr = dt * v
            normdr = xp.sqrt(xp.vdot(dr, dr))
            if normdr > maxstep:
                dr = maxstep * dr / normdr
            pos += dr

        cluster.positions = pos
        return cluster
