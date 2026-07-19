// Bilinear scatter-add for softmax splatting (forward warp), OpenCL C.
//
// One work-item per source pixel (n,y,x); each accumulates the pixel's
// `channels` values into up to 4 destination pixels (bilinear corner
// weights), matching driver/softsplat.py's `_forward_splat` CPU reference
// exactly (same corner-weight formula, same "clamp-or-drop" out-of-bounds
// handling via skip-instead-of-clamp-to-zero-weight -- adding zero is a
// no-op either way, so skipping is equivalent and cheaper).
//
// This AMD driver (RX 7800 XT, OpenCL 2.1) exposes no native float-atomic
// extension (no cl_ext_float_atomics or similar) -- only the standard
// cl_khr_global_int32_base_atomics. Float accumulation is therefore done
// via the portable atomic_cmpxchg CAS-loop-on-reinterpreted-bits idiom.
//
// Softmax weighting (exp(metric)) and post-scatter normalization stay in
// Python (driver/softsplat_cl.py) around this kernel, mirroring how
// driver/softsplat.py structures the CPU version -- this kernel only does
// the core scatter-add over an already-weighted, already-augmented input.

inline void atomic_add_float(volatile __global float *addr, float val) {
    union { unsigned int u32; float f32; } next, expected, current;
    current.f32 = *addr;
    do {
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg((volatile __global unsigned int *)addr,
                                      expected.u32, next.u32);
    } while (current.u32 != expected.u32);
}

__kernel void splat_scatter_add(
    __global const float *source,  // [n_batch, channels, height, width]
    __global const float *flow,    // [n_batch, 2, height, width]
    __global float *out,           // [n_batch, channels, height, width], pre-zeroed
    const int n_batch,
    const int channels,
    const int height,
    const int width)
{
    const int hw = height * width;
    const int gid = get_global_id(0);
    if (gid >= n_batch * hw) return;

    const int n = gid / hw;
    const int rem = gid - n * hw;
    const int y = rem / width;
    const int x = rem - y * width;

    const float flow_x = flow[(n * 2 + 0) * hw + rem];
    const float flow_y = flow[(n * 2 + 1) * hw + rem];
    const float target_x = (float)x + flow_x;
    const float target_y = (float)y + flow_y;
    if (!isfinite(target_x) || !isfinite(target_y)) return;

    const int x0 = (int)floor(target_x);
    const int y0 = (int)floor(target_y);
    const float fx = target_x - (float)x0;
    const float fy = target_y - (float)y0;
    const int src_base = n * channels * hw + rem;

    for (int dy = 0; dy < 2; dy++) {
        const int iy = y0 + dy;
        if (iy < 0 || iy >= height) continue;
        const float wy = dy ? fy : (1.0f - fy);
        for (int dx = 0; dx < 2; dx++) {
            const int ix = x0 + dx;
            if (ix < 0 || ix >= width) continue;
            const float weight = (dx ? fx : (1.0f - fx)) * wy;

            const int dst_base = n * channels * hw + iy * width + ix;
            for (int c = 0; c < channels; c++) {
                atomic_add_float(&out[dst_base + c * hw], source[src_base + c * hw] * weight);
            }
        }
    }
}
