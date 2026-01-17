import numpy as np
from sggo.cluster import Cluster
from sggo.local_opt import LocalOpt


class RandomRestartBaseline:
    def __init__(self, local_optimizer: LocalOpt):
        self.local_optimizer = local_optimizer

    def _energy(self, cluster: Cluster) -> float:
        return float(np.squeeze(self.local_optimizer.energy.energy(cluster)))

    def find_minimum(
        self,
        num_atoms: int,
        num_trials: int,
        *,
        energy_resolution: float = 1e-3,
        target: float | None = None,
        seed: int | None = None,
        return_history: bool = False,
    ):
        best_cluster = None
        best_energy = np.inf

        # optional: track best-so-far
        best_hist = np.empty(num_trials, dtype=float)

        # optional: very light duplicate filtering (same spirit as your GA)
        seen_energies: list[float] = []

        for t in range(num_trials):
            # 1) random init
            cl = Cluster.generate(num_atoms)

            # 2) enforce constraints (if required)
            cl.ensure_seperation()

            # 3) local minimize
            cl_rel = self.local_optimizer.local_min(cl)

            # 4) evaluate
            E = self._energy(cl_rel)

            # optional duplicate suppression
            if any(abs(E - e) < energy_resolution for e in seen_energies):
                best_hist[t] = best_energy
                continue
            seen_energies.append(E)

            # 5) keep best
            if best_energy > E:
                best_energy = E
                best_cluster = cl_rel.copy()

            best_hist[t] = best_energy
            print(f"Trial {t} | best E={best_energy}")

            if target is not None and best_energy <= target:
                best_hist = best_hist[: t + 1]
                break

        if best_cluster is None:
            # In case everything got filtered as duplicate (unlikely), just return last trial
            best_cluster = cl_rel.copy()
            best_energy = E

        if return_history:
            return best_cluster, best_energy, {"best_E": best_hist}
        return best_cluster, best_energy
