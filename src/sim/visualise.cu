#include "sim/kernels.h"
#include "engine/CudaCheck.h"


__global__ void visualiseScalarKernel(const float* field,
    cudaSurfaceObject_t surface,
    int width, int height, float scale)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    //check if im on the grid or not.
    if (x >= width || y >= height) return;

    //read the values.
    const float v = field[y * width + x] * scale;

    //write to the pixels.
    surf2Dwrite(make_float4(v, 0.0f, 0.0f, 1.0f),
        surface, x * static_cast<int>(sizeof(float4)), y);
}

void launchVisualiseScalar(const float* field, cudaSurfaceObject_t surface,
    int width, int height, float scale)
{
    const dim3 block(16, 16);
    const dim3 grid((width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y);

    visualiseScalarKernel << <grid, block >> > (field, surface, width, height, scale);
    CUDA_CHECK_KERNEL();
}