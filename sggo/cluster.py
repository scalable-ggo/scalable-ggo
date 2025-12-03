from numpy.typing import ArrayLike


class Cluster():
    positions: ArrayLike

    def __init__(self, positions: ArrayLike) -> None:
        self.positions = positions

    def copy(self) -> "Cluster":
        return Cluster(self.positions)

    def deepcopy(self) -> "Cluster":
        return Cluster(self.positions.copy())
