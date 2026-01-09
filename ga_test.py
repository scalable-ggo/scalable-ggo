import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from numpy.typing import ArrayLike

from sggo.energy.lj import LJCPU
from sggo.global_opt.genetic import GeneticAlgorithm
from sggo.local_opt.fire import FIRECPU


def idk(x: ArrayLike) -> ArrayLike:
    return x


def idk2() -> float:
    return 0.5


def main():

    num_candidates = 3
    num_atoms = 30

    ga = GeneticAlgorithm(
        num_candidates=num_candidates,
        local_optimizer=FIRECPU(LJCPU()),
        mating_distribution=idk2,
    )

    clusters = [ga.find_minimum(num_atoms, 100)]

    if MPI.COMM_WORLD.rank == 0:
        fig = plt.figure(figsize=(5 * num_candidates, 5))

        for i, coords in enumerate(map(lambda c: np.asarray(c.positions), clusters)):
            ax = fig.add_subplot(1, num_candidates, i + 1, projection="3d")
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])

            ax.set_title(f"Cluster {i + 1}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

            max_range = (coords.max() - coords.min()) / 2.0
            mid_x = (coords[:, 0].max() + coords[:, 0].min()) / 2.0
            mid_y = (coords[:, 1].max() + coords[:, 1].min()) / 2.0
            mid_z = (coords[:, 2].max() + coords[:, 2].min()) / 2.0

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
