#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Every CUDA call returns a status code and almost nobody checks them, which is
// why CUDA bugs usually surface as a black screen three steps later instead of
// at the call that failed. Wrap everything in this.
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t err_ = (expr);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "[cuda] %s:%d  %s\n  -> %s\n",                \
                         __FILE__, __LINE__, #expr, cudaGetErrorString(err_)); \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

// Kernel launches fail asynchronously, so a launch error shows up at the next
// synchronising call unless you explicitly ask. Call this after a launch while
// developing; it costs a sync, so it is compiled out of release builds.
#ifdef NDEBUG
#define CUDA_CHECK_KERNEL() ((void)0)
#else
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)
#endif
