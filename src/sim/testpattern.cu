#include "sim/testpattern.h"
#include "engine/CudaCheck.h"

// ---------------------------------------------------------------------------
// Phase 0 placeholder kernel.
//
// No physics here - this exists purely to prove that a CUDA kernel can write
// into an OpenGL texture. It does however establish the shape that every real
// kernel will take:
//
//   1. work out which grid cell this thread owns
//   2. bail out if that cell is off the edge of the grid
//   3. compute something
//   4. write it
//
// Every operator in the simulation - advection, divergence, the Jacobi
// solver - is that same four-step skeleton with different arithmetic in
// step 3.
// ---------------------------------------------------------------------------

__global__ void testPatternKernel(cudaSurfaceObject_t surface,
                                  int width, int height, float t)
{
    // Step 1: this thread's cell. Each thread handles exactly one cell, and
    // all of them run at once.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Step 2: grids rarely divide evenly into blocks, so the last block runs
    // threads that fall outside the domain. They must not write anything.
    if (x >= width || y >= height) return;

    // Step 3: normalised coordinates in [0,1], then some moving colour.
    const float u = static_cast<float>(x) / static_cast<float>(width);
    const float v = static_cast<float>(y) / static_cast<float>(height);
    const float TAU = 6.28318530718f;

    const float r = 0.5f + 0.5f * sinf(TAU * (u + 0.10f * t));
    const float g = 0.5f + 0.5f * sinf(TAU * (v + 0.13f * t));
    const float b = 0.5f + 0.5f * sinf(TAU * (u + v + 0.07f * t));

    // Step 4: write.
    //
    // GOTCHA: for surface writes the x coordinate is measured in BYTES, not
    // in texels. Forgetting the multiply gives you a stretched image using
    // only the left quarter of the texture - a classic first-day CUDA bug.
    const float4 colour = make_float4(r, g, b, 1.0f);
    surf2Dwrite(colour, surface, x * static_cast<int>(sizeof(float4)), y);
}

void launchTestPattern(cudaSurfaceObject_t surface, int width, int height, float timeSeconds)
{
    // 16x16 = 256 threads per block. A reasonable default for 2D grids: big
    // enough to keep the GPU busy, small enough to stay flexible.
    const dim3 block(16, 16);
    const dim3 grid((width  + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);

    testPatternKernel<<<grid, block>>>(surface, width, height, timeSeconds);
    CUDA_CHECK_KERNEL();
}
