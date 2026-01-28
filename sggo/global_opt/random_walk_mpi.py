from enum import IntEnum

import numpy as np
from mpi4py import MPI

from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt


class RRMPITag(IntEnum):
    TAG_MSG = 1
    TAG_EXIT = 2


class RandomRestart:
    def __init__(self, local_optimizer: LocalOpt):
        self.local_optimizer = local_optimizer

    def find_minimum(self, num_atoms: int, num_epochs: int, target: float | None = None):
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size

        energy_fn = self.local_optimizer.energy.energy

        if rank == 0:
            # controller process
            cluster_current = Cluster.generate(num_atoms)
            cluster_current.ensure_seperation()
            cluster_min = self.local_optimizer.local_min(cluster_current)

            energy_current = float(np.squeeze(energy_fn(cluster_min)))
            energy_min = energy_current

            # give all workers an initial task to do
            for i in range(1, size):
                comm.Send([np.array([0], dtype=np.float32), MPI.FLOAT], dest=i, tag=RRMPITag.TAG_MSG)

            workers = size - 1
            hops = num_epochs - workers

            for step in range(num_epochs):
                if hops == 0 and workers == 0:
                    break

                cluster_opt = None
                energy_opt = None

                if size == 1:
                    # if there is no other processes do work in the main process
                    cluster_new = Cluster.generate(num_atoms)
                    cluster_new.ensure_seperation()

                    cluster_opt = self.local_optimizer.local_min(cluster_new)
                    energy_opt = float(np.squeeze(energy_fn(cluster_opt)))
                else:
                    data = np.zeros(1 + 3 * num_atoms, dtype=np.float32)
                    status = MPI.Status()

                    comm.Recv(data, source=MPI.ANY_SOURCE, tag=RRMPITag.TAG_MSG, status=status)

                    if hops > 0:
                        # assign the next task to the finished process
                        comm.Send([np.zeros(0, dtype=np.float32)],
                                dest=status.Get_source(), tag=RRMPITag.TAG_MSG)
                        hops -= 1
                    else:
                        # ask the process to exit as the desired number of epochs was reached
                        comm.Send([np.zeros(0, dtype=np.float32), MPI.FLOAT],
                                dest=status.Get_source(), tag=RRMPITag.TAG_EXIT)
                        workers -= 1

                    energy_opt = float(data[0])
                    cluster_opt = Cluster(data[1:].reshape(-1, 3))

                if energy_opt < energy_min:
                    cluster_min = cluster_opt
                    energy_min = energy_opt

                print("rr: ", step, energy_opt, energy_min)

                if target is not None and energy_min <= target + 2e-3 and hops > 0:
                    print("Target reached in", step, "epochs")
                    hops = 0

            return energy_min, cluster_min

        else:
            # worker process
            data = np.zeros(0, dtype=np.float32)
            status = MPI.Status()

            while True:
                comm.Recv(data, source=0, tag=MPI.ANY_TAG, status=status)

                if status.Get_tag() == RRMPITag.TAG_EXIT:
                    break

                cluster_new = Cluster.generate(num_atoms)
                cluster_new.ensure_seperation()

                cluster_opt = self.local_optimizer.local_min(cluster_new)
                energy_opt = float(np.squeeze(energy_fn(cluster_opt)))

                comm.Send([np.append(np.float32(energy_opt),
                                    cluster_opt.positions.flatten().astype(np.float32, copy=False)),
                        MPI.FLOAT],
                        dest=0, tag=RRMPITag.TAG_MSG)

            return None, None

