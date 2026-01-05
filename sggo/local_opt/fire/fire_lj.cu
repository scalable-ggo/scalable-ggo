#include <stdint.h>

#define step_reduce vv_reduce
#define step_ratio_shared vf_ratio_shared

extern "C" __global__
void fire_lj(float (* __restrict__ x)[3])
{
    extern __shared__ float pos[1024][3];
    extern __shared__ bool conv_reduce[32];
    extern __shared__ float vf_reduce[32];
    extern __shared__ float ff_reduce[32];
    extern __shared__ float vv_reduce[32];
    extern __shared__ float vf_ratio_shared;

    uint32_t ind = threadIdx.x;
    uint32_t mask = __activemask();
    uint32_t warp_lines = (blockDim.x + 32 - 1) / 32;

    float px = x[ind][0], py = x[ind][1], pz = x[ind][2];
    float vx = 0.f, vy = 0.f, vz = 0.f;

    float dt = DT_START;
    float a = A_START;
    uint32_t step = 0;

    pos[ind][0] = px;
    pos[ind][1] = py;
    pos[ind][2] = pz;

    __syncthreads();

    for (uint32_t itr = 0; itr < MAX_STEPS; itr++) {
        float fx = 0.f, fy = 0.f, fz = 0.f;

        for (uint32_t i = 0; i < blockDim.x; i++) {
            if (i == ind) continue;

            float dx = px - pos[i][0];
            float dy = py - pos[i][1];
            float dz = pz - pos[i][2];

            float c2 = 1.f / (dx * dx + dy * dy + dz * dz);
            float c6 = c2 * c2 * c2;
            float mag = 24.f * (2.f * c6 * c6 - c6) * c2;

            fx += mag * dx;
            fy += mag * dy;
            fz += mag * dz;
        }

        bool conv = fx * fx + fy * fy + fz * fz < TARGET_GRADIENT * TARGET_GRADIENT;
        conv = __all_sync(mask, conv);

        float vf = fx * vx + fy * vy + fz * vz;
        float ff = fx * fx + fy * fy + fz * fz;
        float vv = vx * vx + vy * vy + vz * vz;

        #pragma unroll
        for (uint32_t i = 32 >> 1; i > 0; i >>= 1) {
            vf += __shfl_down_sync(mask, vf, i);
            ff += __shfl_down_sync(mask, ff, i);
            vv += __shfl_down_sync(mask, vv, i);
        }

        if ((ind & (32 - 1)) == 0) {
            conv_reduce[ind / 32] = conv;
            vf_reduce[ind / 32] = vf;
            ff_reduce[ind / 32] = ff;
            vv_reduce[ind / 32] = vv;
        }

        __syncthreads();

        if (ind < warp_lines) {
            uint32_t mask2 = (1u << warp_lines) - 1;
            conv = __all_sync(mask2, conv_reduce[ind]);

            vf = vf_reduce[ind];
            ff = ff_reduce[ind];
            vv = vv_reduce[ind];

            #pragma unroll
            for (uint32_t i = 32 >> 1; i > 0; i >>= 1) {
                vf += __shfl_down_sync(mask2, vf, i);
                ff += __shfl_down_sync(mask2, ff, i);
                vv += __shfl_down_sync(mask2, vv, i);
            }

            if (ind == 0) {
                conv_reduce[0] = conv;
                vf_reduce[0] = vf;
                vf_ratio_shared = sqrtf(vv / ff);
            }
        }

        __syncthreads();

        if (conv_reduce[0])
            break;

        if (vf_reduce[0] > 0.f) {
            float vf_ratio = vf_ratio_shared;
            vx = (1 - a) * vx + a * fx * vf_ratio;
            vy = (1 - a) * vz + a * fy * vf_ratio;
            vz = (1 - a) * vz + a * fz * vf_ratio;

            if (step > NMIN) {
                dt *= FINC;
                dt = dt > DTMAX ? DTMAX : dt;
                a *= FA;
            } else {
                step++;
            }
        } else {
            vx = 0.f, vy = 0.f, vz = 0.f;
            a = A_START;
            dt *= FDEC;
            step = 0;
        }

        vx += fx * dt, vy += fy * dt, vz += fz * dt;
        float dx = vx * dt, dy = vy * dt, dz = vz * dt;

        float step = dx * dx + dy * dy + dz * dz;

        #pragma unroll
        for (uint32_t i = 32 >> 1; i > 0; i >>= 1) {
            step += __shfl_down_sync(mask, step, i);
        }

        if ((ind & (32 - 1)) == 0) {
            step_reduce[ind / 32] = step;
        }

        __syncthreads();

        if (ind < warp_lines) {
            uint32_t mask2 = (1u << warp_lines) - 1;

            step = step_reduce[ind];

            #pragma unroll
            for (uint32_t i = 32 >> 1; i > 0; i >>= 1) {
                step += __shfl_down_sync(mask2, step, i);
            }

            if (ind == 0) {
                step = sqrtf(step);
                step_ratio_shared = step > MAXSTEP ? MAXSTEP / step : 1.f;
            }
        }

        __syncthreads();

        float step_ratio = step_ratio_shared;

        px += dx * step_ratio;
        py += dy * step_ratio;
        pz += dz * step_ratio;

        pos[ind][0] = px;
        pos[ind][1] = py;
        pos[ind][2] = pz;

        __syncthreads();
    }

    x[ind][0] = px;
    x[ind][1] = py;
    x[ind][2] = pz;
}
