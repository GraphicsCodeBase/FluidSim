#include "engine/DisplayTexture.h"
#include "engine/CudaCheck.h"

// glad must come before the CUDA interop header, which expects GL types to
// already be declared.
#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <cstdio>

DisplayTexture::~DisplayTexture()
{
    if (m_surface)  cudaDestroySurfaceObject(m_surface);
    if (m_resource) cudaGraphicsUnregisterResource(m_resource);
    if (m_tex)      glDeleteTextures(1, &m_tex);
}

bool DisplayTexture::create(int width, int height)
{
    m_width  = width;
    m_height = height;

    // --- OpenGL side -------------------------------------------------------
    // RGBA32F because the simulation deals in floats and we do not want the
    // display path quantising values we may want to inspect.
    glCreateTextures(GL_TEXTURE_2D, 1, &m_tex);
    glTextureStorage2D(m_tex, 1, GL_RGBA32F, width, height);
    glTextureParameteri(m_tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(m_tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(m_tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(m_tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // --- Hand it to CUDA ---------------------------------------------------
    // Done once, at startup. The flag says "kernels will write to this via a
    // surface object", which is what lets us use surf2Dwrite below.
    CUDA_CHECK(cudaGraphicsGLRegisterImage(
        &m_resource, m_tex, GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsSurfaceLoadStore));

    std::printf("[interop] registered %dx%d RGBA32F texture with CUDA\n", width, height);
    return true;
}

void DisplayTexture::map()
{
    // Transfer ownership of the texture from OpenGL to CUDA for this frame.
    CUDA_CHECK(cudaGraphicsMapResources(1, &m_resource, 0));

    cudaArray_t array = nullptr;
    CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&array, m_resource, 0, 0));

    // A surface object is the writable view of that array. The mapped array
    // can in principle change between frames, so build the view each time -
    // it costs almost nothing.
    cudaResourceDesc desc{};
    desc.resType         = cudaResourceTypeArray;
    desc.res.array.array = array;
    CUDA_CHECK(cudaCreateSurfaceObject(&m_surface, &desc));
}

void DisplayTexture::unmap()
{
    CUDA_CHECK(cudaDestroySurfaceObject(m_surface));
    m_surface = 0;
    // Give the texture back to OpenGL. Drawing it before this point is
    // undefined behaviour.
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_resource, 0));
}
