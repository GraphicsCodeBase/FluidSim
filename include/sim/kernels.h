#pragma once
#include <cuda_runtime.h>
// ---------------------------------------------------------------------------
// Kernel launchers.
//
// Plain C++ entry points into the CUDA kernels. The kernels themselves stay
// in the .cu files; only these wrappers are visible to the rest of the code.
// ---------------------------------------------------------------------------

// Copy a scalar field into the display texture's red channel.
void launchVisualiseScalar(const float* field, cudaSurfaceObject_t surface,
    int width, int height, float scale);