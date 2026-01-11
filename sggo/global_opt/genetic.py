from enum import IntEnum
from typing import Callable
import numpy as np
from mpi4py import MPI
from numpy.typing import ArrayLike
from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt


class GAMPITag(IntEnum):
    TAG_MSG = 1
    TAG_EXIT = 2


class MutationOperator(IntEnum):
    ANGULAR = 0
    CARTESIAN = 1
    DYNAMIC = 2
    GEOMETRIC_CENTER = 3
    INTERIOR = 4
    RADIAL_SWAP = 5


class GeneticAlgorithm:

    def __init__(self, num_candidates: int, local_optimizer: LocalOpt, mating_distribution: Callable[[], float],
                  operators: list[int | MutationOperator] | None = None):
        self.num_candidates = num_candidates
        self.local_optimizer = local_optimizer
        self.mating_distribution = mating_distribution
    
    def boltzmann_weights(self, energies):
        e = np.asarray(energies, dtype=float).ravel()
        emin = e.min()
        betaE = (e - emin)
        w = np.exp(-betaE)
        w /= w.sum()
        return w
    
    def mutate(self, cluster: Cluster) -> Cluster:
        choice: int
        rng = np.random.default_rng()
        if self.operators is not None and len(self.operators) != 0:
            choice = rng.choice([int(op) for op in self.operators])
        else:
            return cluster
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
                cluster.positions *= rng.uniform(1.0 - gamma, 1.0 + gamma)
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
            # Radial shell swap
            case 5:
                center = cluster.positions.mean(axis=0)
                rel = cluster.positions - center
                r = np.linalg.norm(rel, axis=1)
                idx = np.argsort(r)
                n_core = max(1, N // 4)
                n_surf = max(1, N // 4)

                core_atoms = idx[:n_core]
                surf_atoms = idx[-n_surf:]

                i = rng.choice(core_atoms)
                j = rng.choice(surf_atoms)

                ui = rel[i] / (r[i] + 1e-12)
                uj = rel[j] / (r[j] + 1e-12)

                ri = r[j] * rng.uniform(0.9, 1.1)
                rj = r[i] * rng.uniform(0.9, 1.1)

                cluster.positions[i] = center + ri * ui
                cluster.positions[j] = center + rj * uj
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

    def find_minimum(self, num_atoms: int, num_epochs: int, mutation_rate: float = 0.15,
                      energy_resolution: float = 1e-3, target: float | None = None) -> tuple[Cluster, float]:
        rng = np.random.default_rng()
        energy_fn = self.local_optimizer.energy.energy

        clusters = [Cluster.generate(num_atoms) for _ in range(self.num_candidates)]
        energies = np.empty(self.num_candidates)

        for i, cl in enumerate(clusters):
            cl = self.local_optimizer.local_min(cl)
            clusters[i] = cl
            energies[i] = energy_fn(cl)

        best_idx = int(np.argmin(energies))
        best_cluster = clusters[best_idx].copy()
        best_energy = energies[best_idx]

        k = max(1, self.num_candidates // 5)

        T0 = 1.0
        Tmin = 1e-3

        last_improvement = 0
        IMMIGRATION_DELAY = 10
        IMMIGRATION_FRACTION = 0.1

        for curr_epoch in range(num_epochs):

            T = max(Tmin, T0 * np.exp(-curr_epoch / (0.3 * num_epochs)))
            e = energies
            w = np.exp(-(e - e.min()) / T)
            w /= w.sum()
            for _ in range(k):
                i1 = rng.choice(len(clusters), p=w)
                w[i1] = 0
                w = w / w.sum()
                i2 = rng.choice(len(clusters), p=w)

                parent1 = clusters[i1]
                parent2 = clusters[i2]

                child = self.mate(parent1, parent2)

                p_mut = mutation_rate * np.exp(-curr_epoch / num_epochs)
                if rng.random() < p_mut:
                    child = self.mutate(child)

                child.ensure_seperation()

                child = self.local_optimizer.local_min(child)
                child_energy = energy_fn(child)

                if any(abs(E - child_energy) < energy_resolution for E in energies):
                    if rng.random() < 0.9:
                        continue

                worst_idx = int(np.argmax(energies))
                dE = child_energy - energies[worst_idx]

                if dE < 0 or rng.random() < np.exp(-dE / T):
                    clusters[worst_idx] = child
                    energies[worst_idx] = child_energy

                    if child_energy < best_energy:
                        best_energy = child_energy
                        best_cluster = child.copy()
                        last_improvement = curr_epoch

            if curr_epoch - last_improvement >= IMMIGRATION_DELAY:
                num_immigrants = max(1, int(self.num_candidates * IMMIGRATION_FRACTION))
                worst_indices = np.argsort(energies)[-num_immigrants:]

                for idx in worst_indices:
                    immigrant = Cluster.generate(num_atoms)
                    immigrant = self.local_optimizer.local_min(immigrant)
                    clusters[idx] = immigrant
                    energies[idx] = energy_fn(immigrant)

                last_improvement = curr_epoch

            if target is not None and abs(best_energy - target) <= 1e-3:
                print("Target reached in", curr_epoch, "epochs")
                break

            print(f"Epoch {curr_epoch} | " f"best E={best_energy}")
        return best_cluster, best_energy
