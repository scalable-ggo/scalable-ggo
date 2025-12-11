# from ase import *
# from ase import units
# from ase.visualize import view
# from ase.optimize.basin import BasinHopping
# from ase.calculators.lj import LennardJones
# from ase.optimize import FIRE
# from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

import cProfile
from cupyx.profiler import time_range

from sggo.cluster import Cluster
import sggo.energy.lj as lj
import sggo.local_opt.fire as fire
import sggo.visualize.plot as plot

import cupy as cp
# import numpy as np


@time_range()
def hopBasin(cluster, steps, energy, local_opt):
    cluster_min = local_opt.local_min(cluster)
    energy_min = energy.energy(cluster_min)

    cluster_current = cluster
    energy_current = energy_min

    dr = 0.1
    kT = 100 * 8.617330337217213e-05  # units.kB

    for step in range(steps):
        energy_start = 1e16
        while energy_start > 1e15:
            ro = cluster_current.positions
            rn = ro + dr * cp.random.uniform(-1.0, 1.0, ro.shape, dtype=cp.float32)
            cluster_new = Cluster(rn)
            energy_start = energy.energy(cluster_new)

        cluster_opt = local_opt.local_min(cluster_new)
        energy_opt = energy.energy(cluster_opt)

        if energy_opt < energy_min:
            cluster_min = cluster_opt
            energy_min = energy_opt

        accept = cp.exp((energy_current - energy_opt) / kT) > cp.random.uniform()
        if accept:
            cluster_current = cluster_new
            energy_current = energy_opt

        print("basin: ", step, energy_opt, energy_min)

    return energy_min, cluster_min


energy = lj.create()
local_opt = fire.create(energy)

n = 1024
r = n / 4

cluster = None
En = 1e16
while En > 1e15:
    pos = cp.random.uniform(-1, 1, (n, 3), dtype=cp.float32)
    pos /= cp.sqrt((pos * pos).sum(1))[:, cp.newaxis]
    pos *= cp.cbrt(cp.random.uniform(0, r, n))[:, cp.newaxis]
    cluster = Cluster(pos)

    En = energy.energy(cluster)


emin, clustermin = None, None
cProfile.run("emin, clustermin = hopBasin(cluster, 2000, energy, local_opt)", sort=1)
posmin = cp.asnumpy(clustermin.positions)

plot = plot.ClusterPlot(clustermin)
plot.plot()

for p in posmin:
    print(p)

# atomlist = []

# for i in range(n):
#     atomlist.append(Atom('He', posmin[i]))
# print(emin)
# view(Atoms(atomlist))

# atomlist = []
#
# for i in range(5):
#     for j in range(5):
#         for k in range(5):
#             atomlist.append(Atom('He', (1 * i, 1 * j, 1 * k)))
#
# system = Atoms(atomlist, calculator = LennardJones(rc = np.inf))
#
# bh = BasinHopping(atoms=system, optimizer=FIRE, logfile="-", trajectory=None, optimizer_logfile=None, local_minima_trajectory=None, adjust_cm=False)
#
# cProfile.run("bh.run(10)", sort=1)
# _, optimal = bh.get_minimum()
# view(optimal)
