from typing import Tuple
import numpy as np

from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt
from sggo.types import NDArray


class BasinHopping:
    def __init__(self, local_optimizer: LocalOpt):
        self.local_optimizer = local_optimizer

    def mutate(self, cluster: Cluster, dr: float) -> Cluster:
        ro = cluster.positions
        rn = ro + dr * np.random.uniform(-1.0, 1.0, ro.shape).astype(np.float32)
        cluster_new = Cluster(rn)
        cluster_new.ensure_seperation()

        return cluster_new

    def find_minimum(self, cluster_start: Cluster, num_epochs: int) -> Tuple[NDArray, Cluster]:
        energy_fn = self.local_optimizer.energy.energy

        cluster_min = self.local_optimizer.local_min(cluster_start)
        energy_min = energy_fn(cluster_min)

        cluster_current = cluster_start
        energy_current = energy_min

        dr = 0.1
        kT = 100 * 8.617330337217213e-05  # units.kB

        for step in range(num_epochs):
            cluster_new = self.mutate(cluster_current, dr)

            cluster_opt = self.local_optimizer.local_min(cluster_new)
            energy_opt = energy_fn(cluster_opt)

            if energy_opt < energy_min:
                cluster_min = cluster_opt
                energy_min = energy_opt

            accept = np.exp((energy_current - energy_opt) / kT) > np.random.uniform()
            if accept:
                cluster_current = cluster_new
                energy_current = energy_opt

            print("basin: ", step, energy_opt, energy_min)

        return energy_min, cluster_min
