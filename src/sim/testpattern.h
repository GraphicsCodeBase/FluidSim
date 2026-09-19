#pragma once

#include <cuda_runtime.h>

// Phase 0 placeholder. Writes an animated pattern straight into the display
// texture so we can prove the CUDA -> OpenGL path works before any physics
// exists. Deleted in Phase 1, when real fields replace it.
void launchTestPattern(cudaSurfaceObject_t surface, int width, int height, float timeSeconds);
