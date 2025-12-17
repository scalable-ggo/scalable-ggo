import cProfile

import numpy as np

from sggo.cluster import Cluster
from sggo.energy import lj
from sggo.local_opt import bfgs, cg, fire
from sggo.visualize import plot


def hop_basin(cluster, steps, energy, local_opt):
    cluster_min = local_opt.local_min(cluster)
    energy_min = energy.energy(cluster_min)

    cluster_current = cluster
    energy_current = energy_min

    dr = 0.1
    kT = 100 * 8.617330337217213e-05  # units.kB

    for step in range(steps):
        ro = cluster_current.positions
        rn = ro + dr * np.random.uniform(-1.0, 1.0, ro.shape).astype(np.float32)
        cluster_new = Cluster(rn)
        cluster_new.ensure_seperation()

        cluster_opt = local_opt.local_min(cluster_new)
        energy_opt = energy.energy(cluster_opt)

        if energy_opt < energy_min:
            cluster_min = cluster_opt
            energy_min = energy_opt

        accept = np.exp((energy_current - energy_opt) / kT) > np.random.uniform()
        if accept:
            cluster_current = cluster_new
            energy_current = energy_opt

        print("basin: ", step, energy_opt, energy_min)

    return energy_min, cluster_min


energy = lj.create()
local_opt = fire.create(energy)
cluster = Cluster.generate(128)

emin, clustermin = None, None
cProfile.run("emin, clustermin = hop_basin(cluster, 50, energy, local_opt)", sort=1)

print("Energy: ", emin)
plot = plot.ClusterPlot(clustermin)
plot.plot()
