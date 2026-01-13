from enum import IntEnum

import numpy as np
from mpi4py import MPI

from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt


class BHMPITag(IntEnum):
    TAG_MSG = 1
    TAG_EXIT = 2


class BasinHopping:
    def __init__(self, local_optimizer: LocalOpt):
        self.local_optimizer = local_optimizer

    def mutate(self, cluster: Cluster, dr: float) -> Cluster:
        ro = cluster.positions
        rn = ro + dr * np.random.uniform(-1.0, 1.0, ro.shape).astype(np.float32)
        cluster_new = Cluster(rn)
        cluster_new.ensure_seperation()

        return cluster_new

    def find_minimum(self, num_atoms: int, num_epochs: int, target: float | None = None) -> Cluster:
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size

        energy_fn = self.local_optimizer.energy.energy
        dr = 0.1

        if rank == 0:
            # controller process
            cluster_current = Cluster.generate(num_atoms)
            cluster_min = self.local_optimizer.local_min(cluster_current)

            energy_current = energy_fn(cluster_min)
            energy_min = energy_current

            kT = 100 * 8.617330337217213e-05  # units.kB

            # give all workers an initial task to do
            for i in range(1, size):
                comm.Send([cluster_current.positions.flatten(), MPI.FLOAT], dest=i, tag=BHMPITag.TAG_MSG)

            workers = size - 1
            hops = num_epochs - workers

            for step in range(num_epochs):
                if hops == 0 and workers == 0:
                    break

                cluster_new = None
                cluster_opt = None
                energy_opt = None

                if size == 1:
                    # if there is no other processes do work in the main process
                    cluster_new = self.mutate(cluster_current, dr)

                    cluster_opt = self.local_optimizer.local_min(cluster_new)
                    energy_opt = energy_fn(cluster_opt)
                else:
                    data = np.zeros(2 * 3 * num_atoms + 1, dtype=np.float32)
                    status = MPI.Status()

                    comm.Recv(data, source=MPI.ANY_SOURCE, tag=BHMPITag.TAG_MSG, status=status)
                    if hops > 0:
                        # assign the next task to the finished process
                        comm.Send([cluster_current.positions.flatten(), MPI.FLOAT],
                                  dest=status.Get_source(), tag=BHMPITag.TAG_MSG)
                        hops -= 1
                    else:
                        # ask the process to exit as the desired number of epochs was reached
                        comm.Send([np.zeros(0), MPI.FLOAT], dest=status.Get_source(), tag=BHMPITag.TAG_EXIT)
                        workers -= 1

                    cluster_new = Cluster(data[1:3 * num_atoms + 1].reshape(-1, 3))
                    cluster_opt = Cluster(data[3 * num_atoms + 1:].reshape(-1, 3))
                    energy_opt = data[0]

                if energy_opt < energy_min:
                    cluster_min = cluster_opt
                    energy_min = energy_opt

                accept = np.exp((energy_current - energy_opt) / kT) > np.random.uniform()
                if accept:
                    cluster_current = cluster_new
                    energy_current = energy_opt

                print("basin: ", step, energy_opt, energy_min)

                if target is not None and energy_min <= target + 2e-3 and hops > 0:
                    print("Target reached in", step, "epochs")
                    hops = 0

            return energy_min, cluster_min
        else:
            # worker process
            data = np.zeros(3 * num_atoms, dtype=np.float32)
            status = MPI.Status()

            while True:
                comm.Recv(data, source=0, tag=MPI.ANY_TAG, status=status)

                if status.Get_tag() == BHMPITag.TAG_EXIT:
                    break

                cluster_current = Cluster(data.reshape(-1, 3))
                cluster_new = self.mutate(cluster_current, dr)

                cluster_opt = self.local_optimizer.local_min(cluster_new)
                energy_opt = energy_fn(cluster_opt)

                comm.Send([np.append(energy_opt, (
                    cluster_new.positions.flatten(),
                    cluster_opt.positions.flatten())), MPI.FLOAT], dest=0, tag=BHMPITag.TAG_MSG)

            return None, None
