from collections import namedtuple
from numpy.typing import ArrayLike
from typing import Callable, Optional

from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt

ClusterSplit = namedtuple("ClusterSplit", ["normal", "lower", "upper"])


class GeneticAlgorithm:
    def __init__(self, num_candidates: int, local_optimizer: LocalOpt, mating_distribution: Callable[[], float]):
        self.num_candidates = num_candidates
        self.local_optimizer = local_optimizer
        self.mating_distribution = mating_distribution

    def create_clusters(self, num_atoms: int) -> list[Cluster]:
        raise [Cluster.generate(num_atoms) for _ in range(self.num_candidates)]

    def mutate(self, cluster: Cluster) -> Cluster:
        raise NotImplementedError()

    def split(self, cluster: Cluster, *, plane_normal: Optional[ArrayLike] = None) -> ClusterSplit:
        raise NotImplementedError()

    def join(self, cluster1: ArrayLike, cluster2: ArrayLike) -> Cluster:
        raise NotImplementedError()

    def mate(self, cluster1: Cluster, cluster2: Cluster) -> Cluster:
        split1 = self.split(cluster1)
        split2 = self.split(cluster1, plane_normal=split1.normal)
        return self.join(split1.lower, split2.upper)

    def find_minimum(self, num_atoms: int, num_epochs: int) -> Cluster:
        raise NotImplementedError()
