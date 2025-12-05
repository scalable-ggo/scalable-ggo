from numpy.typing import ArrayLike
from sggo.energy import Energy

import cupy as cp


class LJ(Energy):
    def _energies_shared(self, pos: ArrayLike) -> ArrayLike:
        xp = cp.get_array_module(pos)

        disp = pos[:, xp.newaxis] - pos

        r2 = (disp * disp).sum(2)
        xp.fill_diagonal(r2, xp.inf)
        c2 = xp.reciprocal(r2)
        c6 = c2 * c2 * c2
        c12 = c6 * c6

        return (xp.float32(2) * (c12 - c6)).sum(1)

    def _energy_shared(self, pos: ArrayLike) -> float:
        xp = cp.get_array_module(pos)

        return xp.sum(self._energies_shared(pos))

    def _energy_gradient_shared(self, pos: ArrayLike) -> ArrayLike:
        xp = cp.get_array_module(pos)

        disp = pos[:, xp.newaxis] - pos

        r2 = (disp * disp).sum(2)
        xp.fill_diagonal(r2, xp.inf)
        c2 = xp.reciprocal(r2)
        c6 = c2 * c2 * c2
        c12 = c6 * c6

        force_mags = xp.float32(24) * (xp.float32(2) * c12 - c6) * c2

        return (force_mags[:, :, xp.newaxis] * disp).sum(1)
