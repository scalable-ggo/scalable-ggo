from sggo.global_opt.genetic import GeneticAlgorithm
from sggo.local_opt.fire import FIRECPU
from sggo.energy.lj import LJCPU
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt


def idk(x: ArrayLike) -> ArrayLike:
    return x


def idk2() -> float:
    return 0.5


def main():

    num_candidates = 10     
    num_atoms = 60
    R = 5.0                 
    for i in [0,1,2,3,4]:
        ga = GeneticAlgorithm(
        num_candidates=num_candidates,
        local_optimizer=FIRECPU(LJCPU()),
        mating_distribution=idk2,
        r=R,
        operators=[j for j in range(0,i)]
    )
        clusters, energy = ga.find_minimum(num_atoms, 100)
        print(energy)


if __name__ == "__main__":
    main()
