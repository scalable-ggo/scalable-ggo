import numpy as np
from typing import Callable

from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt
from numpy.typing import ArrayLike
from numpy.typing import ArrayLike


class GeneticAlgorithm:
    def __init__(self, num_candidates: int, local_optimizer: LocalOpt, mating_distribution: Callable[[], float], r: float):
        self.num_candidates = num_candidates
        self.local_optimizer = local_optimizer
        self.mating_distribution = mating_distribution
        self.r = r
    
    def boltzmann_weights(self, energies):
        e = np.array(energies)
        emin = e.min()
        betaE = (e - emin)
        w = np.exp(-betaE)
        w /= w.sum()
        return w
    
    def mutate(self, cluster: Cluster) -> Cluster:
        rng = np.random.default_rng()
        choice: int = rng.integers(0, 5)
        N: int = cluster.positions.shape[0]

        # the distance to the nearest neighbour of the ith atom
        def nn_dist(i: int) -> float:
            diff: ArrayLike = cluster.positions - cluster.positions[i]
            dists: ArrayLike = np.linalg.norm(diff, axis=1)
            dists[i] = np.inf
            return np.min(dists)

        match choice:
            # Angular operator
            case 0:
                noAtoms: int = rng.integers(1, max(2, int(N / 20) + 1)) # 1-5%, otherwise take 1 as default
                chosenAtoms: ArrayLike = rng.choice(N, size=noAtoms, replace=False)
                center: ArrayLike = np.mean(cluster.positions, axis=0)
                for i in chosenAtoms:
                    radius: float = np.linalg.norm(cluster.positions[i] - center)
                    direction: ArrayLike = rng.normal(size=3)
                    direction /= np.linalg.norm(direction)
                    cluster.positions[i] = center + radius * direction
            # Cartesian Displacement Operator
            case 1:
                S: float = 0.2
                noAtoms: int = rng.integers(1, N + 1)
                chosenAtoms: ArrayLike = rng.choice(N, size=noAtoms, replace=False)
                for i in chosenAtoms:
                    rmin: float = nn_dist(i)
                    displacement: ArrayLike = rng.uniform(-1.0, 1.0, size=3)
                    cluster.positions[i] += (S * rmin) * displacement
            # Dynamic Mutation
            case 2:
                gamma: float = 0.10
                cluster.positions *= rng.uniform(1.0 - gamma, 1.0 + gamma, size=(N, 3))
            # Geometric Center Displacement Operator
            case 3:
                amax: float = 0.2
                amin: float = 0.7
                w: float = 2.0

                center: ArrayLike = np.mean(cluster.positions, axis=0)
                rMax: float = np.max(np.linalg.norm(cluster.positions - center, axis=1))

                noAtoms: int = rng.integers(1, N + 1)
                chosenAtoms: ArrayLike = rng.choice(N, size=noAtoms, replace=False)

                for i in chosenAtoms:
                    ri: float = np.linalg.norm(cluster.positions[i] - center)
                    rmin: float = nn_dist(i)
                    direction: ArrayLike = rng.normal(size=3)
                    direction /= np.linalg.norm(direction)
                    cluster.positions[i] += ((amax - amin) * (ri / rMax)**w + amin) * rmin * direction
            # Interior Operator
            case 4:
                atom_index: int = rng.integers(0, N)
                center: ArrayLike = np.mean(cluster.positions, axis=0)
                ri: float = np.linalg.norm(cluster.positions[atom_index] - center)
                radius: float = rng.uniform(0.01, 0.10) * ri
                direction: ArrayLike = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                cluster.positions[atom_index] = center + radius * direction

        return cluster

    def mate(self, cluster1: Cluster, cluster2: Cluster) -> Cluster:
        # translate the clusters so that their centers are at (0, 0, 0)
        p1 = cluster1.positions - np.mean(cluster1.positions, axis=0)
        p2 = cluster2.positions - np.mean(cluster2.positions, axis=0)
        # choose a random plane to slice the clusters, represented by a normal vector and (implicitly) point (0, 0, 0)
        plane_normal = np.random.uniform(low=-1.0, high=1.0, size=3)
        plane_normal /= np.linalg.norm(plane_normal)
        # calculate the distances from each atom to the plane
        d1 = np.dot(p1, plane_normal)
        d2 = np.dot(p2, plane_normal)
        # argsort instead of sort cause the indices will be used later to sort p1 and p2
        idx1 = np.argsort(d1)
        idx2 = np.argsort(d2)
        d1 = d1[idx1]
        d2 = d2[idx2]
        # translate the parents if necessary
        i = d1.searchsorted(0)
        j = d2.searchsorted(0)
        while i < j:
            if abs(d1[i]) < abs(d2[j - 1]):
                i += 1
            else:
                j -= 1
        while i > j:
            if abs(d1[i - 1]) < abs(d2[j + 1]):
                i -= 1
            else:
                j += 1
        return Cluster(np.concatenate([p1[idx1[:i]], p2[idx2[i:]]]))

    def find_minimum(self, num_atoms: int, num_epochs: int, mutation_rate: float = 0.05, energy_resolution: float = 1e-3) -> Cluster:
        rng = np.random.default_rng()
        energy_fn = self.local_optimizer.energy.energy

        clusters = [Cluster.generate(num_atoms, energy_fn, self.r) for _ in range(self.num_candidates)]
        energies = []

        for i, cl in enumerate(clusters):
            relaxed = self.local_optimizer.local_min(cl)
            E = energy_fn(relaxed)
            energies.append(E)
            clusters[i] = relaxed  

        best_idx = int(np.argmin(energies))
        best_cluster = clusters[best_idx].copy()
        best_energy = energies[best_idx]

        for _ in range(num_epochs):
            weights = self.boltzmann_weights(energies)
            i1 = rng.choice(len(clusters), p=weights)
            i2 = rng.choice(len(clusters), p=weights)
            while i2 == i1:
                i2 = rng.choice(len(clusters), p=weights)

            parent1 = clusters[i1]
            parent2 = clusters[i2]

            child = self.mate(parent1, parent2)

            if rng.random() < mutation_rate:
                child = self.mutate(child)

            child_relaxed = self.local_optimizer.local_min(child)
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
