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
        self.num_candidates = num_candidates
        self.local_optimizer = local_optimizer
        self.mating_distribution = mating_distribution
    
    def create_clusters(self, num_atoms: int) -> list[ArrayLike]:
        raise NotImplementedError()
    
    def mutate(self, cluster: ArrayLike) -> ArrayLike:
        rng = np.random.default_rng()
        choice: int = rng.integers(0, 5)
        N: int = cluster.shape[0]

        # the distance to the nearest neighbour of the ith atom
        def nn_dist(i: int) -> float:
            diff: ArrayLike = cluster - cluster[i]
            dists: ArrayLike = np.linalg.norm(diff, axis=1)
            dists[i] = np.inf
            return np.min(dists)

        match choice:
            # Angular operator
            case 0:
                noAtoms: int = rng.integers(1, max(2, int(N/20) + 1)) # 1-5%, otherwise take 1 as default
                chosenAtoms: ArrayLike = rng.choice(N, size=noAtoms, replace=False)
                center: ArrayLike = np.mean(cluster, axis=0)
                for i in chosenAtoms:
                    radius: float = np.linalg.norm(cluster[i] - center)
                    direction: ArrayLike = rng.normal(size=3)
                    direction = direction / np.linalg.norm(direction)
                    cluster[i] = center + radius * direction
            # Cartesian Displacement Operator
            case 1:
                S: float = 0.2
                noAtoms: int = rng.integers(1, N + 1)
                chosenAtoms: ArrayLike = rng.choice(N, size=noAtoms, replace=False)
                for i in chosenAtoms:
                    rmin: float = nn_dist(i)
                    displacement: ArrayLike = rng.uniform(-1.0, 1.0, size=3)
                    cluster[i] = cluster[i] + (S * rmin) * displacement
            # Dynamic Mutation
            case 2:
                gamma: float = 0.10
                cluster = cluster * rng.uniform(1.0 - gamma, 1.0 + gamma, size=(N, 3))
            # Geometric Center Displacement Operator
            case 3:
                amax: float = 0.2
                amin: float = 0.7
                w: float = 2.0

                center: ArrayLike = np.mean(cluster, axis=0)
                rMax: float = np.max(np.linalg.norm(cluster - center, axis=1))

                noAtoms: int = rng.integers(1, N + 1)
                chosenAtoms: ArrayLike = rng.choice(N, size=noAtoms, replace=False)

                for i in chosenAtoms:
                    ri: float = np.linalg.norm(cluster[i] - center)
                    rmin: float = nn_dist(i)
                    direction: ArrayLike = rng.normal(size=3)
                    direction = direction / np.linalg.norm(direction)
                    cluster[i] = cluster[i] + ((amax - amin) * (ri / rMax)**w + amin) * rmin * direction
            # Interior Operator
            case 4:
                atom_index: int = rng.integers(0, N)
                center: ArrayLike = np.mean(cluster, axis=0)
                ri: float = np.linalg.norm(cluster[atom_index] - center)
                radius: float = rng.uniform(0.01, 0.10) * ri
                direction: ArrayLike = rng.normal(size=3)
                direction = direction / np.linalg.norm(direction)
                cluster[atom_index] = center + radius * direction

        return cluster
    
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
    
