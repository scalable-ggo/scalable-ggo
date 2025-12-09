from numpy.typing import ArrayLike
from typing import Callable
import numpy as np


class GeneticAlgorithm:
    def __init__(self, num_candidates: int, local_optimizer: Callable[[ArrayLike], ArrayLike], mating_distribution: Callable[[], float], R: float):
        self.num_candidates = num_candidates
        self.local_optimizer = local_optimizer
        self.mating_distribution = mating_distribution
        self.R = R
    
    def create_clusters(self, num_atoms: int) -> list[ArrayLike]:
         num_candidates = self.num_candidates
         clusters: list[ArrayLike] = []
         R = self.R
         for i in range(num_candidates):

            coords = []

            for _ in range(num_atoms):
                v = np.random.normal(size=3)
                length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
                v = v / length 

                u = np.random.random()
                r = R * (u ** (1/3))

                x = v[0] * r
                y = v[1] * r
                z = v[2] * r

                coords.append([x, y, z])

            clusters.append(np.array(coords))

         return clusters
    
    def boltzmann_weights(energies):
        e = np.array(energies)
        emin = e.min()
        betaE = (e - emin)
        w = np.exp(-betaE)
        w /= w.sum()
        return w
        
    
    def mutate(self, cluster: ArrayLike) -> ArrayLike:
        raise NotImplementedError()
    
    def split(self, cluster: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        raise NotImplementedError()

    def join(self, cluster1: ArrayLike, cluster2: ArrayLike) -> ArrayLike:
        raise NotImplementedError()


    def find_minimum(self, num_atoms: int, num_epochs: int, energy_fn: Callable[[ArrayLike], float], mutation_rate: float = 0.05, energy_resolution: float = 1e-3) -> ArrayLike:

        rng = np.random.default_rng()

        clusters = self.create_clusters(num_atoms)
        energies = []

        for cl in clusters:
            relaxed = self.local_optimizer(cl)
            E = energy_fn(relaxed)
            energies.append(E)
            clusters[clusters.index(cl)] = relaxed  

        best_idx = int(np.argmin(energies))
        best_cluster = clusters[best_idx].copy()
        best_energy = energies[best_idx]

        for i in range(num_epochs):

            weights = self.boltzmann_weights(energies)
            i1 = rng.choice(len(clusters), p=weights)
            i2 = rng.choice(len(clusters), p=weights)
            while i2 == i1:
                i2 = rng.choice(len(clusters), p=weights)

            parent1 = clusters[i1]
            parent2 = clusters[i2]

            split1 = self.split(parent1)
            split2 = self.split(parent2, plane_normal=split1.normal)


            child = self.join(left1, right2)

            if rng.random() < mutation_rate:
                child = self.mutate(child)

            child_relaxed = self.local_optimizer(child)
            child_energy = energy_fn(child_relaxed)

            duplicate = False
            for E in energies:
                if abs(E - child_energy) < energy_resolution:
                    duplicate = True
                    break
            if duplicate:
                continue

            worst_idx = int(np.argmax(energies))
            if child_energy < energies[worst_idx]:
                clusters[worst_idx] = child_relaxed
                energies[worst_idx] = child_energy

                if child_energy < best_energy:
                    best_energy = child_energy
                    best_cluster = child_relaxed.copy()

        return best_cluster
    
