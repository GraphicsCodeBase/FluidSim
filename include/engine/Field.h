#pragma once

#include "engine/CudaCheck.h"

#include <cuda_runtime.h>
#include <utility>   // std::swap

// ---------------------------------------------------------------------------
// A simulation grid, double-buffered.
//
// Every operator in the solver reads a cell's four neighbours, so no kernel
// can write into the buffer it is reading - it would clobber values other
// threads still need. Each Field therefore owns two allocations: kernels read
// from one and write to the other, then swap() makes the result current.
//
//   advectKernel<<<...>>>(dye.read(), dye.write(), w, h, dt);
//   dye.swap();
// ---------------------------------------------------------------------------
template <typename T>
class Field
{
public:
    Field() = default;
    ~Field() { release(); }

    // Owns GPU memory, so copying it would free the same pointer twice.
    Field(const Field&) = delete;
    Field& operator=(const Field&) = delete;

    void create(int width, int height)
    {
        release();
        m_width = width;
        m_height = height;
        CUDA_CHECK(cudaMalloc(&m_read, bytes()));
        CUDA_CHECK(cudaMalloc(&m_write, bytes()));
        clear();
    }

    void release()
    {
        if (m_read)  cudaFree(m_read);
        if (m_write) cudaFree(m_write);
        m_read = m_write = nullptr;
    }

    void clear()
    {
        CUDA_CHECK(cudaMemset(m_read, 0, bytes()));
        CUDA_CHECK(cudaMemset(m_write, 0, bytes()));
    }

    // Kernels read from here...
    const T* read() const { return m_read; }
    // ...and write to here.
    T* write() { return m_write; }

    // Promote what a kernel just wrote to be the current contents.
    void swap() { std::swap(m_read, m_write); }

    int    width()  const { return m_width; }
    int    height() const { return m_height; }
    int    count()  const { return m_width * m_height; }
    size_t bytes()  const { return static_cast<size_t>(count()) * sizeof(T); }

private:
    T* m_read = nullptr;
    T* m_write = nullptr;
    int m_width = 0;
    int m_height = 0;
};

using ScalarField = Field<float>;    // pressure, divergence, dye
using VectorField = Field<float2>;   // velocity