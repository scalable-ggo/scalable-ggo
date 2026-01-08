from sggo.global_opt.genetic import GeneticAlgorithm
from sggo.local_opt.fire import FIRECPU
from sggo.local_opt.fire import FIREGPU
from sggo.energy.lj import LJGPU
from sggo.energy.lj import LJCPU
from sggo.energy.lj import LJGPUKernel
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt


def main():

    num_candidates = 50     
    num_atoms = 300
    ga = GeneticAlgorithm(
                num_candidates=num_candidates,
                local_optimizer=FIREGPU(LJGPU()),
                mating_distribution= 0.5,
                operators=[1,2,3,4,5]
            )
    clusters, energy = ga.find_minimum(num_atoms, 2000, target = -1942.106775)
    print(energy)

if __name__ == "__main__":
    main()
