from numpy.typing import ArrayLike
from typing import Callable


class GeneticAlgorithm:
    def __init__(self, num_candidates: int, local_optimizer: Callable[[ArrayLike], ArrayLike], mating_distribution: Callable[[], float]):
        self.num_candidates = num_candidates
        self.local_optimizer = local_optimizer
        self.mating_distribution = mating_distribution
    
    def create_clusters(self, num_atoms: int) -> list[ArrayLike]:
        raise NotImplementedError()
    
    def mutate(self, cluster: ArrayLike) -> ArrayLike:
        raise NotImplementedError()
    
    def split(self, cluster: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        raise NotImplementedError()

    def join(self, cluster1: ArrayLike, cluster2: ArrayLike) -> ArrayLike:
        raise NotImplementedError()
    
    def find_minimum(self, num_atoms: int, num_epochs: int) -> ArrayLike:
        raise NotImplementedError()
