import numpy as np
from typing import Callable

from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt


class GeneticAlgorithm:
    def __init__(self, num_candidates: int, local_optimizer: LocalOpt, mating_distribution: Callable[[], float], r: float):
        self.num_candidates = num_candidates
        self.local_optimizer = local_optimizer
        self.mating_distribution = mating_distribution
        self.r = r
    
    def create_clusters(self, num_atoms: int) -> list[Cluster]:
        num_candidates = self.num_candidates
        clusters: list[Cluster] = []
        r = self.r
        for i in range(num_candidates):

            coords = []

            for j in range(num_atoms):
                v = np.random.normal(size=3)
                length = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
                v /= length 

                u = np.random.random()
                r *= (u ** (1 / 3))

                x = v[0] * r
                y = v[1] * r
                z = v[2] * r

                coords.append([x, y, z])

            clusters.append(Cluster(np.array(coords)))

        return clusters
    
    def boltzmann_weights(self, energies):
        e = np.array(energies)
        emin = e.min()
        betaE = (e - emin)
        w = np.exp(-betaE)
        w /= w.sum()
        return w

    def mutate(self, cluster: Cluster) -> Cluster:
        raise NotImplementedError()

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
        return Cluster(np.concat([p1[idx1[:i]], p2[idx2[i:]]]))

    def find_minimum(self, num_atoms: int, num_epochs: int, mutation_rate: float = 0.05, energy_resolution: float = 1e-3) -> Cluster:
        rng = np.random.default_rng()
        energy_fn = self.local_optimizer.energy.energy

        clusters = self.create_clusters(num_atoms)
        energies = []

        for i, cl in enumerate(clusters):
            relaxed = self.local_optimizer(cl)
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
    
