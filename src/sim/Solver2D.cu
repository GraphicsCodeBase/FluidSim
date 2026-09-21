#include "sim/Solver2D.h"

#include <cstdio>

void Solver2D::create(int width, int height)
{
    // Store the dimensions first. Nothing below strictly needs them yet, but
    // everything after this does - and a grid width left at zero would make
    // every later kernel launch silently do nothing at all.
    m_width  = width;
    m_height = height;

    // Two allocations each: Field is double-buffered, because every operator
    // reads its neighbours and so cannot write into the buffer it is reading.
    m_u.create(width, height);
    m_dye.create(width, height);

    std::printf("[solver] %dx%d grid, %d cells\n", width, height, m_u.count());
    std::printf("[solver] velocity %zu bytes/buffer, dye %zu bytes/buffer\n",
                m_u.bytes(), m_dye.bytes());
}
