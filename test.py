# from ase import *
# from ase import units
# from ase.visualize import view
# from ase.optimize.basin import BasinHopping
# from ase.calculators.lj import LennardJones
# from ase.optimize import FIRE
# from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG

import cProfile
from cupyx.profiler import time_range

import math
import cupy as cp
# import numpy as np

warpsize = 32

pairwise_energy_kernel = cp.RawKernel(r'''
#include <stdint.h>

extern "C" __global__
void pairwise_energy(const uint32_t n, const float4 * __restrict__ x, float * __restrict__ y) {
    extern __shared__ float sums[];

    uint32_t id1 = blockIdx.x;
    uint32_t id2 = threadIdx.x;

    float energy;

    if (id2 < n && id1 != id2) {
        float4 pos1 = x[id1];
        float4 pos2 = x[id2];

        float dx = pos1.x - pos2.x;
        float dy = pos1.y - pos2.y;
        float dz = pos1.z - pos2.z;

        float c2 = 1.f / (dx * dx + dy * dy + dz * dz);
        float c6 = c2 * c2 * c2;

        energy = 4.f * (c6 * c6 - c6);
    } else {
        energy = 0.0;
    }

    #pragma unroll
    for (uint32_t i = warpSize >> 1; i > 0; i >>= 1) {
        energy += __shfl_down_sync(~0, energy, i);
    }

    if ((id2 & (warpSize - 1)) == 0) {
        uint32_t warpid = id2 / warpSize;
        sums[warpid] = energy;
    }

    __syncthreads();
    uint32_t warplines = blockDim.x / warpSize;

    if (id2 < warplines) {
        uint32_t mask = (1u<<warplines)    - 1;

        energy = sums[id2];

        #pragma unroll
        for (uint32_t i = warplines >> 1; i > 0; i >>= 1) {
            energy += __shfl_down_sync(mask, energy, i);
        }
    }

    if (id2 == 0) {
        y[id1] = energy;
    }
}
''', 'pairwise_energy', backend="nvcc")

pairwise_force_kernel = cp.RawKernel(r'''
#include <stdint.h>

extern "C" __global__
void pairwise_force(const uint32_t n, const float4 * __restrict__ x, float4 * __restrict__ y) {
    extern __shared__ float4 sums[];

    uint32_t id1 = blockIdx.x;
    uint32_t id2 = threadIdx.x;

    float4 force;

    if (id2 < n && id1 != id2) {
        float4 pos1 = x[id1];
        float4 pos2 = x[id2];

        float dx = pos1.x - pos2.x;
        float dy = pos1.y - pos2.y;
        float dz = pos1.z - pos2.z;

        float c2 = 1.f / (dx * dx + dy * dy + dz * dz);
        float c6 = c2 * c2 * c2;
        float mag = 24.f * (2.f * c6 * c6  - c6) * c2;

        force.x = mag * dx;
        force.y = mag * dy;
        force.z = mag * dz;
        force.w = 0.f;
    } else {
        force.x = 0.f;
        force.y = 0.f;
        force.z = 0.f;
        force.w = 0.f;
    }

    #pragma unroll
    for (uint32_t i = warpSize >> 1; i > 0; i >>= 1) {
        force.x += __shfl_down_sync(~0, force.x, i);
        force.y += __shfl_down_sync(~0, force.y, i);
        force.z += __shfl_down_sync(~0, force.z, i);
    }

    if ((id2 & (warpSize - 1)) == 0) {
        uint32_t warpid = id2 / warpSize;
        sums[warpid] = force;
    }

    __syncthreads();
    uint32_t warplines = blockDim.x / warpSize;

    if (id2 < warplines) {
        uint32_t mask = (1u<<warplines)    - 1;

        force = sums[id2];

        #pragma unroll
        for (uint32_t i = warplines >> 1; i > 0; i >>= 1) {
            force.x += __shfl_down_sync(mask, force.x, i);
            force.y += __shfl_down_sync(mask, force.y, i);
            force.z += __shfl_down_sync(mask, force.z, i);
        }
    }

    if (id2 == 0) {
        y[id1] = force;
    }
}
''', 'pairwise_force', backend="nvcc")

pairwise_energy = None
@time_range()
def get_energy(pos):
    n = len(pos)
    global pairwise_energy # noqa: PLW0603
    if pairwise_energy is None:
        pairwise_energy = cp.zeros((n), dtype=cp.float32)

    warplines = math.ceil(n / warpsize)
    pairwise_energy_kernel((n,), (warplines * warpsize,), (n, pos, pairwise_energy), shared_mem=warplines*4)

    return cp.sum(pairwise_energy) / 2

# @time_range()
# def get_energy(pos):
#     disp = pos[:,cp.newaxis] - pos
#
#     r2 = (disp * disp).sum(2)
#     cp.fill_diagonal(r2, cp.inf)
#     c2 = cp.reciprocal(r2)
#     c6 = c2 * c2 * c2
#     c12 = c6 * c6
#
#     return cp.sum(cp.float32(2) * (c12 - c6))

pairwise_force = None
@time_range()
def get_forces(pos):
    n = len(pos)
    global pairwise_force # noqa: PLW0603
    if pairwise_force is None:
        pairwise_force = cp.zeros((n, 4), dtype=cp.float32)

    warplines = math.ceil(n / warpsize)
    pairwise_force_kernel((n,), (warplines * warpsize,), (n, pos, pairwise_force), shared_mem=4*warplines*4)

    return pairwise_force

# @time_range()
# def get_forces(pos):
#     disp = pos[:,cp.newaxis] - pos
#
#     r2 = (disp * disp).sum(2)
#     cp.fill_diagonal(r2, cp.inf)
#     c2 = cp.reciprocal(r2)
#     c6 = c2 * c2 * c2
#     c12 = c6 * c6
#
#     force_mags = cp.float32(24) * (cp.float32(2) * c12 - c6) * c2
#
#     return (force_mags[:, :, cp.newaxis] * disp).sum(1)


@time_range()
def fire(pos):
    max_steps = 100000
    maxstep = 0.2
    fmax = 0.1
    dt = 0.1
    dtmax = 1.0
    Nmin = 5
    finc = 1.1
    fdec = 0.5
    aStart = 0.1
    fa = 0.99

    a = aStart
    v = None
    step_cnt = 0
    pos = pos.copy()

    for _ in range(max_steps):
        f = get_forces(pos)
        norm = cp.linalg.norm(f, axis=1).max()
        if norm < fmax:
            break

        if v is None:
            v = cp.zeros_like(pos, dtype=cp.float32)
        else:
            vf = cp.vdot(f, v)
            if vf > 0.0:
                v = (1.0 - a) * v + a * f / cp.sqrt(
                    cp.vdot(f, f)) * cp.sqrt(cp.vdot(v, v))
                if step_cnt > Nmin:
                    dt = min(dt * finc, dtmax)
                    a *= fa
                step_cnt += 1
            else:
                v *= 0.0
                a = aStart
                dt *= fdec
                step_cnt = 0

        v += dt * f
        dr = dt * v
        normdr = cp.sqrt(cp.vdot(dr, dr))
        if normdr > maxstep:
            dr = maxstep * dr / normdr
        pos += dr

    return get_energy(pos), pos

@time_range()
def hopBasin(pos, steps):
    pos = cp.pad(pos, ((0, 0), (0, 1)))
    Emin, rmin = fire(pos)
    ro = pos
    Eo = Emin
    dr = 0.1
    kT = 100 * 8.617330337217213e-05 # units.kB

    for step in range(steps):
        En = 1e16
        while En > 1e15:
            rn = ro + dr * cp.random.uniform(-1., 1., ro.shape, dtype=cp.float32)
            rn[:, 3] = 0.0
            En = get_energy(rn)
        En, ropt = fire(rn)

        if En < Emin:
            rmin = ropt
            Emin = En

        accept = cp.exp((Eo - En) / kT) > cp.random.uniform()
        if accept:
            ro = rn
            Eo = En

        print("basin: ", step, En, Emin)

    return Emin, rmin[:, 0:3]

n = 1024
r = n / 4

pos = None
En = 1e16
while En > 1e15:
    pos = cp.random.uniform(-1, 1, (n, 3), dtype=cp.float32)
    pos /= cp.sqrt((pos * pos).sum(1))[:,cp.newaxis]
    pos *= cp.cbrt(cp.random.uniform(0, r, n))[:,cp.newaxis]

    En = get_energy(pos)

cProfile.run("emin, posmin = hopBasin(pos, 2000)", sort=1)
posmin = posmin.get()  # noqa: F821

for p in posmin:
    print(p)

# atomlist = []

# for i in range(n):
#     atomlist.append(Atom('He', posmin[i]))
# print(emin)
# view(Atoms(atomlist))

# atomlist = []
#
# for i in range(5):
#     for j in range(5):
#         for k in range(5):
#             atomlist.append(Atom('He', (1 * i, 1 * j, 1 * k)))
#
# system = Atoms(atomlist, calculator = LennardJones(rc = np.inf))
#
# bh = BasinHopping(atoms=system, optimizer=FIRE, logfile="-", trajectory=None, optimizer_logfile=None, local_minima_trajectory=None, adjust_cm=False)
#
# cProfile.run("bh.run(10)", sort=1)
# _, optimal = bh.get_minimum()
# view(optimal)
