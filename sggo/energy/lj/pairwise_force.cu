#include <stdint.h>

extern "C" __global__
void pairwise_force(const uint32_t n, const float (* __restrict__ x)[3], float (* __restrict__ y)[3]) {
    extern __shared__ float sums[][3];

    uint32_t id1 = blockIdx.x;
    uint32_t id2 = threadIdx.x;

    float fx;
    float fy;
    float fz;

    if (id2 < n && id1 != id2) {
        float dx = x[id1][0] - x[id2][0];
        float dy = x[id1][1] - x[id2][1];
        float dz = x[id1][2] - x[id2][2];

        float c2 = 1.f / (dx * dx + dy * dy + dz * dz);
        float c6 = c2 * c2 * c2;
        float mag = -24.f * (2.f * c6 * c6  - c6) * c2;

        fx = mag * dx;
        fy = mag * dy;
        fz = mag * dz;
    } else {
        fx = 0.f;
        fy = 0.f;
        fz = 0.f;
    }

    #pragma unroll
    for (uint32_t i = warpSize >> 1; i > 0; i >>= 1) {
        fx += __shfl_down_sync(~0, fx, i);
        fy += __shfl_down_sync(~0, fy, i);
        fz += __shfl_down_sync(~0, fz, i);
    }

    if ((id2 & (warpSize - 1)) == 0) {
        uint32_t warpid = id2 / warpSize;
        sums[warpid][0] = fx;
        sums[warpid][1] = fy;
        sums[warpid][2] = fz;
    }

    __syncthreads();
    uint32_t warplines = blockDim.x / warpSize;

    if (id2 < warplines) {
        uint32_t mask = (1u<<warplines)    - 1;

        fx = sums[id2][0];
        fy = sums[id2][1];
        fz = sums[id2][2];

        #pragma unroll
        for (uint32_t i = warplines >> 1; i > 0; i >>= 1) {
            fx += __shfl_down_sync(mask, fx, i);
            fy += __shfl_down_sync(mask, fy, i);
            fz += __shfl_down_sync(mask, fz, i);
        }
    }

    if (id2 == 0) {
        y[id1][0] = fx;
        y[id1][1] = fy;
        y[id1][2] = fz;
    }
}
