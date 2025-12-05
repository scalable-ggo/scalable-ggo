import numpy as np
from typing import Callable

from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt


class GeneticAlgorithm:
    def __init__(self, num_candidates: int, local_optimizer: LocalOpt, mating_distribution: Callable[[], float]):
        self.num_candidates = num_candidates
        self.local_optimizer = local_optimizer
        self.mating_distribution = mating_distribution

    def create_clusters(self, num_atoms: int) -> list[Cluster]:
        raise [Cluster.generate(num_atoms) for _ in range(self.num_candidates)]

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

    def find_minimum(self, num_atoms: int, num_epochs: int) -> Cluster:
        raise NotImplementedError()
