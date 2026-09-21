#pragma once

#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// The CUDA <-> OpenGL bridge. This is the only place in the entire project
// where the two APIs meet, and it is worth reading once even though it is
// Layer 1 scaffolding.
//
// The idea: allocate a texture with OpenGL, hand it to CUDA once at startup,
// and thereafter let CUDA kernels write into it directly. The pixels never
// travel to the CPU and back - they stay in VRAM the whole time.
//
// Note that only the *display* texture needs this treatment. Velocity,
// pressure and divergence live in plain cudaMalloc'd memory, because OpenGL
// never needs to see them.
// ---------------------------------------------------------------------------
class DisplayTexture
{
public:
    ~DisplayTexture();

    bool create(int width, int height);

    // Hand the texture to CUDA for the duration of the simulation step.
    // OpenGL must not touch it between map() and unmap().
    void map();
    void unmap();

    // Valid only between map() and unmap(). This is what kernels write to.
    cudaSurfaceObject_t surface() const { return m_surface; }

    // Valid only outside map()/unmap(). This is what the fragment shader samples.
    unsigned int glTexture() const { return m_tex; }

    int width()  const { return m_width; }
    int height() const { return m_height; }

private:
    unsigned int           m_tex      = 0;
    cudaGraphicsResource_t m_resource = nullptr;
    cudaSurfaceObject_t    m_surface  = 0;
    int m_width  = 0;
    int m_height = 0;
};
