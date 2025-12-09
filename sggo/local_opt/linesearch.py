from numpy.typing import ArrayLike

from sggo.cluster import Cluster
from sggo.energy import Energy

import cupy as cp


def linesearch_wolfe(cluster: Cluster, energy: Energy, direction: ArrayLike, c1: float, c2: float) -> float:
    xp = cp.get_array_module(cluster.positions, direction)
    search_cluster = cluster.copy()

    c1 = xp.float32(c1)
    c2 = xp.float32(c2)

    value0 = energy.energy(search_cluster)
    gradient0 = xp.vdot(energy.energy_gradient(search_cluster), direction)

    a_prev, a_cur = xp.float32(0), 1 / xp.linalg.norm(direction)
    value_prev, value_cur = value0, None
    gradient = None

    a_lo, a_hi = None, None

    while True:
        search_cluster.positions = cluster.positions + a_cur * direction
        value_cur = energy.energy(search_cluster)

        if value_cur > value0 + c1 * a_cur * gradient0 or value_cur >= value_prev and a_prev > xp.float32(0):
            value_lo, a_lo, a_hi = value_prev, a_prev, a_cur
            break

        gradient = xp.vdot(energy.energy_gradient(search_cluster), direction)

        if xp.abs(gradient) <= -c2 * gradient0:
            return a_cur
        if gradient >= xp.float32(0):
            value_lo, a_lo, a_hi = value_cur, a_cur, a_prev
            break

        value_prev = value_cur
        a_prev, a_cur = a_cur, xp.float32(2) * a_cur

    for _ in range(10):
        a_mid = (a_lo + a_hi) / xp.float32(2)
        search_cluster.positions = cluster.positions + a_mid * direction
        value_mid = energy.energy(search_cluster)

        if value_mid > value0 + c1 * a_mid * gradient0 or value_mid > value_lo:
            a_hi = a_mid
        else:
            gradient_mid = xp.vdot(energy.energy_gradient(search_cluster), direction)

            if xp.abs(gradient_mid) <= -c2 * gradient0:
                return a_mid

            if gradient_mid * (a_hi - a_lo) >= xp.float32(0):
                a_hi = a_lo
            a_lo, value_lo = a_mid, value_mid

    return a_mid
