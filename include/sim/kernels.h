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

// Inject dye and a velocity impulse in a Gaussian blob around (posX, posY).
// The external force term f - the only source of energy in the simulation.
void launchSplat(const float2* uIn, float2* uOut,
    const float* dyeIn, float* dyeOut,
    int width, int height,
    float posX, float posY,
    float impulseX, float impulseY,
    float dyeAmount, float radius);