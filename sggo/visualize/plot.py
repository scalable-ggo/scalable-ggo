import matplotlib.pyplot as plt
import numpy as np

from sggo.cluster import Cluster


class ClusterPlot:
    def __init__(self, cluster: Cluster, atom_color: str = "#80C9E4", size: int = 200) -> None:
        self.cluster: Cluster = cluster
        self.N: int = cluster.positions.shape[0]
        self.atom_color: str = atom_color
        self.size: int = size

    def plot(self) -> None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.set_title(f"Cluster — {self.N} atoms", fontsize=16, pad=20)

        positions = np.asarray(self.cluster.positions.get())

        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            s=self.size,
            c=self.atom_color,
            edgecolors="black",
            linewidth=0.3,
            alpha=0.9,
        )

        ax.set_box_aspect([1, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.set_facecolor("white")

        plt.show()
