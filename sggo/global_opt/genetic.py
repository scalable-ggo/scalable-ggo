from enum import IntEnum
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

    def __init__(self, num_candidates: int, local_optimizer: LocalOpt, operators: list[int | MutationOperator] | None = None):
        self.num_candidates = num_candidates
        self.local_optimizer = local_optimizer
        self.operators = operators

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

    def boltzmann_choice(self, rng, temp, energies, clusters):
        e = np.asarray(energies, dtype=float).ravel()
        emin = e.min()
        betaE = (e - emin)
        w = np.exp(-betaE / temp)
        w /= w.sum()

        inds = rng.choice(len(energies), p=w, size=2, replace=False)
        return clusters[inds[0]], clusters[inds[1]]

    def find_minimum(self, num_atoms: int, num_epochs: int, mutation_rate: float = 0.15,
                      energy_resolution: float = 1e-3, target: float | None = None) -> tuple[Cluster, float]:
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size

        rng = np.random.default_rng()
        energy_fn = lambda cluster: np.squeeze(self.local_optimizer.energy.energy(cluster))

        def create_child(parent1, parent2, mutation_rate):
            child = self.mate(parent1, parent2)

            if rng.random() < mutation_rate:
                child = self.mutate(child)

            child.ensure_seperation()

            child_relaxed = self.local_optimizer.local_min(child)
            child_energy = energy_fn(child_relaxed)

            return child_energy, child_relaxed

        def create_immigrant():
            immigrant = Cluster.generate(num_atoms)
            immigrant_relaxed = self.local_optimizer.local_min(immigrant)
            immigrant_energy = energy_fn(immigrant_relaxed)

            return immigrant_energy, immigrant_relaxed

        if rank == 0:
            # controller process
            clusters = []
            energies = []

            for _ in range(self.num_candidates):
                energy, cluster = create_immigrant()

                clusters.append(cluster)
                energies.append(energy)

            best_idx = int(np.argmin(energies))
            best_cluster = clusters[best_idx].copy()
            best_energy = energies[best_idx]

            T0 = 1.0
            Tmin = 1e-3

            last_improvement = 0
            IMMIGRATION_DELAY = 10 * max(1, self.num_candidates // 5)
            IMMIGRATION_FRACTION = 0.1

            # give all workers an initial task to do
            for i in range(1, size):
                parent1, parent2 = self.boltzmann_choice(rng, T0, energies, clusters)
                comm.Send([np.append(parent1.positions.flatten(), parent2.positions.flatten()), MPI.FLOAT], dest=i, tag=GAMPITag.TAG_MSG)

            workers = size - 1
            matings = num_epochs - workers

            for curr_epoch in range(num_epochs):
                if matings == 0 and workers == 0:
                    break
                print(f"Epoch {curr_epoch} | " f"best E={best_energy}")

                T = max(Tmin, T0 * np.exp(-curr_epoch / (0.3 * num_epochs)))
                parent1, parent2 = self.boltzmann_choice(rng, T, energies, clusters)

                child_relaxed = None
                child_energy = None

                p_mut = np.float32(mutation_rate * np.exp(-curr_epoch / num_epochs))

                if size == 1:
                    # if there is no other processes create a child in the main process
                    child_energy, child_relaxed = create_child(parent1, parent2, p_mut)
                else:
                    data = np.zeros(3 * num_atoms + 1, dtype=np.float32)
                    status = MPI.Status()

                    comm.Recv(data, source=MPI.ANY_SOURCE, tag=GAMPITag.TAG_MSG, status=status) # recieve any children finished
                    if matings > 0:
                        # assign the next mating task to the finished process
                        send_data = np.concatenate((
                            np.array([p_mut]),
                            parent1.positions.flatten(),
                            parent2.positions.flatten()
                        ))
                        comm.Send([send_data, MPI.FLOAT], dest=status.Get_source(), tag=GAMPITag.TAG_MSG)
                        matings -= 1
                    else:
                        # ask the process to exit as the desired number of epochs was reached
                        comm.Send([np.zeros(0), MPI.FLOAT], dest=status.Get_source(), tag=GAMPITag.TAG_EXIT)
                        workers -= 1

                    child_relaxed = Cluster(data[1:].reshape(-1, 3))
                    child_energy = data[0]

                duplicate = False
                for E in energies:
                    if abs(E - child_energy) < energy_resolution:
                        duplicate = True
                        break
                if duplicate and rng.random() < 0.9:
                    continue

                worst_idx = int(np.argmax(energies))
                dE = child_energy - energies[worst_idx]

                if dE < 0 or rng.random() < np.exp(-dE / T):
                    clusters[worst_idx] = child_relaxed
                    energies[worst_idx] = child_energy

                    if child_energy < best_energy:
                        best_energy = child_energy
                        best_cluster = child_relaxed.copy()
                        last_improvement = curr_epoch

                if curr_epoch - last_improvement >= IMMIGRATION_DELAY:
                    num_immigrants = max(1, int(self.num_candidates * IMMIGRATION_FRACTION))
                    worst_indices = np.argsort(energies)[-num_immigrants:]

                    for idx in worst_indices:
                        energy, cluster = create_immigrant()
                        clusters[idx] = cluster
                        energies[idx] = energy

                    last_improvement = curr_epoch

                if target is not None and best_energy <= target and matings > 0:
                    print("Target reached in", curr_epoch, "epochs")
                    matings = 0

            return best_energy, best_cluster
        else:
            # worker process
            data = np.zeros(2 * 3 * num_atoms + 1, dtype=np.float32)
            status = MPI.Status()

            while True:
                comm.Recv(data, source=0, tag=MPI.ANY_TAG, status=status)

                if status.Get_tag() == GAMPITag.TAG_EXIT:
                    break

                p_mut = data[0]
                parent1 = Cluster(data[1:3 * num_atoms + 1].reshape(-1, 3))
                parent2 = Cluster(data[3 * num_atoms + 1:].reshape(-1, 3))

                child_energy, child_relaxed = create_child(parent1, parent2, p_mut)
                comm.Send([np.append(child_energy, child_relaxed.positions.flatten()), MPI.FLOAT], dest=0, tag=GAMPITag.TAG_MSG)
            return None, None
