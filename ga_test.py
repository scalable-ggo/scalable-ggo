from sggo.global_opt.genetic import GeneticAlgorithm
from sggo.local_opt.fire import FIRECPU
from sggo.local_opt.fire import FIREGPU
from sggo.energy.lj import LJGPU
from sggo.energy.lj import LJCPU
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt


def idk(x: ArrayLike) -> ArrayLike:
    return x


def idk2() -> float:
    return 0.5


def main():

    num_candidates = 50     
    num_atoms = 98
    energies = np.zeros(6)   
    for i in [5]:
        for _ in range(0,10):
            ga = GeneticAlgorithm(
                num_candidates=num_candidates,
                local_optimizer=FIRECPU(LJCPU()),
                mating_distribution=idk2,
                operators=[0,1,2,3,4]
            )
            clusters, energy = ga.find_minimum(num_atoms, 1000)
            energies[i] +=energy
            print(energy)
    energies /=10
    print(energies)

if __name__ == "__main__":
    main()
