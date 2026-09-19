# FluidSim

Real-time smoke simulation on the GPU — incompressible Navier–Stokes solved with CUDA, displayed with OpenGL.

`C++17` · `CUDA 13` · `OpenGL 4.5` · `CMake + vcpkg`

---

## What this is

A from-scratch GPU implementation of Jos Stam's *Stable Fluids*, following the two NVIDIA GPU Gems chapters on fluid simulation (see [References](#references)).

The goal is not just to produce convincing smoke. It's to take the continuum mechanics — a coupled system of partial differential equations — and turn it into correct, measurably accurate, real-time code. So alongside the visuals, this project carries a validation harness: analytic-solution error measurement, grid-refinement convergence studies, and live divergence/conservation diagnostics.

Smoke rather than liquid, deliberately. Gas has no free surface, so the whole problem stays focused on the Navier–Stokes solver itself rather than on surface-tracking machinery.

## Status

**Phase 0 complete** — CUDA writes directly into an OpenGL texture with no CPU round-trip, at 165 fps using ~5% of the frame budget.

| Phase | Description | Status |
|---|---|:--:|
| 0 | CUDA ↔ OpenGL interop, window, ImGui | ✅ |
| 1 | 2D solver — advection, projection, boundaries | ⬜ |
| 2 | Validation — Taylor–Green, convergence, diagnostics | ⬜ |
| 3 | 3D grids, buoyancy, vorticity confinement | ⬜ |
| 4 | Volume rendering with single-scattering shadows | ⬜ |
| 5 | Performance — Nsight, red–black GS, multigrid | ⬜ |
| 6 | Obstacles, MacCormack advection, combustion | ⬜ |

See [PLAN.md](PLAN.md) for the full project plan, feasibility analysis, and hardware budgets.

## The algorithm

Incompressible Navier–Stokes:

```
∂u/∂t = −(u·∇)u − (1/ρ)∇p + ν∇²u + f
∇·u = 0
```

This isn't solved directly. Following Stam, the terms are **operator split** and advanced in sequence each frame — `u ← P ∘ F ∘ D ∘ A(u)`:

| Step | Term | Method |
|---|---|---|
| **A**dvect | `−(u·∇)u` | Semi-Lagrangian backtrace — unconditionally stable |
| **D**iffuse | `ν∇²u` | Implicit Jacobi (≈0 for smoke) |
| **F**orce | `f` | Explicit Euler — buoyancy, user input |
| **P**roject | `−∇p` | Solve `∇²p = ∇·u`, then `u ← u − ∇p` |

Projection is the interesting part: it enforces incompressibility via the Helmholtz–Hodge decomposition, requires a Poisson solve every frame, and accounts for roughly 80% of the runtime.

## Requirements

| | |
|---|---|
| GPU | NVIDIA, compute capability ≥ 7.5 (developed on an RTX 2070 SUPER) |
| CUDA | Toolkit 13.0 |
| Compiler | Visual Studio 2022 (MSVC v143) |
| CMake | ≥ 3.21 |
| vcpkg | any recent checkout |

Dependencies (`glfw3`, `glad`, `glm`, `imgui`) are declared in `vcpkg.json` and pinned to a baseline commit, so the build is reproducible.

## Build

```bash
cmake --preset default
cmake --build --preset release
```

`CMakePresets.json` hardcodes the vcpkg toolchain at `C:/vcpkg`. If your checkout lives elsewhere, edit `CMAKE_TOOLCHAIN_FILE` there.

Debug builds enable `CUDA_CHECK_KERNEL()`, which synchronises after every kernel launch and reports errors at the failing launch rather than several steps later:

```bash
cmake --build --preset debug
```

## Run

```bash
./build/bin/fluid.exe
```

`Esc` or the window close button exits.

## Layout

The source is split into three layers. The dependency direction is strictly one-way — **physics knows nothing about rendering**, and the simulation would run unchanged with the entire observation layer deleted.

```
src/
├── main.cpp
├── engine/          Layer 1 — Foundation (scaffolding)
│   ├── App          window, GL context, ImGui, frame loop
│   ├── Shader       GLSL load / compile / link
│   ├── DisplayTexture   the single CUDA ↔ OpenGL bridge
│   └── CudaCheck.h  error-checking macros
├── sim/             Layer 2 — Physics (the actual project)
│   └── testpattern.cu   Phase 0 placeholder
└── shaders/         Layer 3 — Observation
    ├── fullscreen.vert  fullscreen triangle from gl_VertexID
    └── display.frag
```

Only `DisplayTexture` touches both CUDA and OpenGL. Velocity, pressure and divergence live in plain `cudaMalloc`'d memory — OpenGL never sees them, so there is exactly one `cudaGraphicsGLRegisterImage` call in the project.

## Development notes

- **Shaders hot-reload.** `.glsl` files are read from disk at runtime via the `SHADER_DIR` compile definition — edit and restart, no rebuild.
- **`nvcc` defaults to C++14**, silently diverging from the host compiler. `CMAKE_CUDA_STANDARD 17` is set explicitly to prevent this.
- **Surface writes index in bytes, not texels.** `surf2Dwrite(v, surf, x * sizeof(float4), y)`. Omitting the multiply is the classic first-day interop bug.
- **Grid `+y` points up** (texture row 0 samples at the bottom of the screen), so buoyancy needs no sign flips later.
- Both Debug and Release write to `build/bin/fluid.exe`.

## References

1. **Stam, J. (1999).** *Stable Fluids.* SIGGRAPH. — The source. Read first; both chapters below are GPU ports of it.
2. **Harris, M.** [*Fast Fluid Dynamics Simulation on the GPU.*](https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu) GPU Gems, ch. 38. — The 2D algorithm, with the clearest pseudocode.
3. **Crane, K., Llamas, I., Tariq, S.** [*Real-Time Simulation and Rendering of 3D Fluids.*](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids) GPU Gems 3, ch. 30. — 3D, buoyancy, vorticity confinement, volume rendering.
4. **Bridson, R.** *Fluid Simulation for Computer Graphics.* — The reference book for boundary conditions and the pressure solve.

## License

Not yet chosen — add a `LICENSE` file before making this repository public.
