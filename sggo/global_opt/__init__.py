from abc import ABC, abstractmethod
from typing import Tuple

from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt


class GlobalOpt(ABC):
    local_opt: LocalOpt

    def __init__(self, local_opt: LocalOpt):
        self.local_opt = local_opt

    @abstractmethod
    def find_minimum(self, num_atoms: int, num_epochs: int, target: float | None = None) -> Tuple[float, Cluster]:
        raise NotImplementedError()
