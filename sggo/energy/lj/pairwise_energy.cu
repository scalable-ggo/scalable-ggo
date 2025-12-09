#include <stdint.h>

extern "C" __global__
void pairwise_energy(const uint32_t n, const float (* __restrict__ x)[3], float * __restrict__ y) {
    extern __shared__ float sums[];

    uint32_t id1 = blockIdx.x;
    uint32_t id2 = threadIdx.x;

    float energy;

    if (id2 < n && id1 != id2) {
        float dx = x[id1][0] - x[id2][0];
        float dy = x[id1][1] - x[id2][1];
        float dz = x[id1][2] - x[id2][2];

        float c2 = 1.f / (dx * dx + dy * dy + dz * dz);
        float c6 = c2 * c2 * c2;

        energy = 2.f * (c6 * c6 - c6);
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
