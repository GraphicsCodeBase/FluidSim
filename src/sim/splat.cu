#include "sim/kernels.h"
#include "engine/CudaCheck.h"

// ---------------------------------------------------------------------------
// Splat - inject dye and momentum around a point.
//
// This is the external force term f in the momentum equation:
//
//     du/dt = -(u.grad)u - grad(p)/rho + nu*lap(u) + f
//                                                    ^
//
// It is the only place energy ever enters the simulation. Every other
// operator - advection, projection, vorticity confinement - only moves around
// or corrects what this kernel puts in. Nothing else creates anything.
//
// Each cell scales its contribution by how far it is from the splat centre,
// using a Gaussian falloff:
//
//     weight = exp( -d^2 / r^2 )
//
// which is 1.0 at the centre, ~0.37 at one radius out, and ~0.018 at two.
//
// Why a Gaussian and not a hard-edged disc: a hard edge is a discontinuity in
// the velocity field, and the pressure solve cannot resolve one cleanly - you
// would see a blocky artifact sit there for several frames after every click.
// A Gaussian is smooth everywhere, so the solver has nothing to fight.
//
// Note this kernel reads no neighbours - each cell needs only its own value
// and its distance to the centre. It could therefore safely write in place.
// We keep the read -> write -> swap pattern anyway so that every operator in
// the solver has the same shape, and because advection genuinely requires it.
// ---------------------------------------------------------------------------
__global__ void splatKernel(const float2* uIn, float2* uOut,
    const float* dyeIn, float* dyeOut,
    int width, int height,
    float px, float py,
    float impulseX, float impulseY,
    float dyeAmount, float radius)
{
    // Which cell does this thread own? Every one of the ~147k threads runs
    // these two lines and each gets a different answer.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Blocks are 16x16 and the grid rarely divides evenly by 16, so the last
    // row and column of blocks launch threads that fall outside the domain.
    // They must not touch memory.
    if (x >= width || y >= height) return;

    // The grid is a flat array, so a 2D coordinate becomes a 1D offset:
    // skip down y whole rows, then across x.
    const int idx = y * width + x;

    // Offset from this cell to the splat centre, in grid cells.
    const float dx = static_cast<float>(x) - px;
    const float dy = static_cast<float>(y) - py;

    // The Gaussian. Note there is no sqrtf here: we want d^2, and
    // (dx*dx + dy*dy) already is d^2 - taking a square root only to square it
    // again would be wasted work.
    //
    // expf rather than exp: the f suffix is the single-precision version.
    // Plain exp works on doubles, which run at 1/32 speed on this GPU.
    const float weight = expf(-(dx * dx + dy * dy) / (radius * radius));

    // Velocity: push the fluid in the direction the mouse is moving.
    //
    // The += matters. Velocity is state - it holds whatever earlier frames
    // put there. Assigning instead of adding would erase all existing motion
    // on every splat, and the fluid would have no memory.
    float2 u = uIn[idx];
    u.x += impulseX * weight;
    u.y += impulseY * weight;
    uOut[idx] = u;

    // Dye: one number, because it has no direction. It is purely a passenger -
    // visible, but with no effect at all on the physics. Delete it and the
    // simulation would run identically; you just could not see anything.
    dyeOut[idx] = dyeIn[idx] + dyeAmount * weight;

    // Every cell writes, including distant ones where weight is ~0. Those
    // simply copy their old value across. That is what stops the back buffer
    // holding stale data when the caller swaps.
}

// ---------------------------------------------------------------------------
// The launcher: ordinary C++, runs once on the CPU. Its only job is to work
// out how many threads to start, then fire the kernel. This is the name the
// rest of the program knows - splatKernel never leaves this file.
// ---------------------------------------------------------------------------
void launchSplat(const float2* uIn, float2* uOut,
                 const float* dyeIn, float* dyeOut,
                 int width, int height,
                 float posX, float posY,
                 float impulseX, float impulseY,
                 float dyeAmount, float radius)
{
    // 256 threads per block. A multiple of 32 (the warp size) keeps the
    // hardware fed; 16x16 is a well-worn default for 2D grids.
    const dim3 block(16, 16);

    // How many blocks to cover the whole domain? Round UP, because integer
    // division truncates: a 500-wide grid needs 32 blocks, not 31, or the
    // last 4 columns would never be processed at all.
    //
    // Rounding up means the last block overhangs the grid edge. The bounds
    // check at the top of the kernel is what discards that overhang - the
    // two are designed together and neither works alone.
    const dim3 grid((width  + block.x - 1) / block.x,
                    (height + block.y - 1) / block.y);

    splatKernel<<<grid, block>>>(uIn, uOut, dyeIn, dyeOut,
                                 width, height, posX, posY,
                                 impulseX, impulseY, dyeAmount, radius);

    // Aborts at the failing launch rather than letting the error surface
    // three steps later as an unexplained black screen. Compiles to nothing
    // in Release builds.
    CUDA_CHECK_KERNEL();
}
