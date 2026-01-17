from mpi4py import MPI

from sggo.energy import lj
from sggo.global_opt.random_walk_mpi import RandomRestart  
from sggo.local_opt import fire
from sggo.visualize.plot import ClusterPlot


def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    energy = lj.create()
    local_opt = fire.create(energy)

    rr = RandomRestart(local_optimizer=local_opt)

    best_energy, best_cluster = rr.find_minimum(
        num_atoms=38,
        num_epochs=10000,
        target=-173.928427,
    )

    if rank == 0:
        print(best_energy)

        plot = ClusterPlot(best_cluster)
        plot.plot()


if __name__ == "__main__":
    main()
