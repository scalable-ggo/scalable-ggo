import cupy as cp

from sggo.energy import Energy
from sggo.types import NDArray


class LJ(Energy):
    def _energies_shared(self, pos: NDArray) -> NDArray:
        xp = cp.get_array_module(pos)

        disp = pos[:, xp.newaxis] - pos

        r2 = (disp * disp).sum(2)
        xp.fill_diagonal(r2, xp.inf)
        c2 = xp.reciprocal(r2)
        c6 = c2 * c2 * c2
        c12 = c6 * c6

        return (xp.float32(2) * (c12 - c6)).sum(1)

    def _energy_shared(self, pos: NDArray) -> float:
        xp = cp.get_array_module(pos)

        return xp.sum(self._energies_shared(pos))

    def _energy_gradient_shared(self, pos: NDArray) -> NDArray:
        xp = cp.get_array_module(pos)

        disp = pos[:, xp.newaxis] - pos

        r2 = (disp * disp).sum(2)
        xp.fill_diagonal(r2, xp.inf)
        c2 = xp.reciprocal(r2)
        c6 = c2 * c2 * c2
        c12 = c6 * c6

        mags = xp.float32(-24) * (xp.float32(2) * c12 - c6) * c2

        return (mags[:, :, xp.newaxis] * disp).sum(1)
