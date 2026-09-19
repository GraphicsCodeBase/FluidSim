# GPU Smoke & Fluid Simulation — Project Plan

**Goal:** demonstrate the ability to take heavy continuum-mechanics theory and turn it into a correct, performant, interactive GPU application — with evidence of understanding, not just a pretty video.

**Stack:** C++17 · CUDA 13 · OpenGL 4.5 (CUDA–GL interop) · CMake
**Cadence:** evenings + weekends, no deadline.

---

## 0. Verified environment

| Component | Status |
|---|---|
| GPU | RTX 2070 SUPER, 8 GB, compute capability **sm_75** (supported by CUDA 13) |
| CUDA | 13.0.88 (`nvcc`) |
| Host compiler | MSVC 14.44.35207 (VS2022 Community) — the required host compiler on Windows |
| CMake | 3.23.1 — **verified working** with CUDA 13; no upgrade needed |
| vcpkg | `C:/vcpkg`, bootstrapped, baseline `7824193` (2025-11-19) |
| Python 3.9 | available — use for the validation/plotting harness |

### Setup status: cleared

Both snags I originally flagged turned out to be non-issues:

1. **CMake 3.23 handles CUDA 13 fine.** It identifies `NVIDIA 13.0.88`, resolves `CUDAToolkit`, and compiles for `sm_75` without complaint. Verified by building and running a real kernel. No upgrade required.
2. **vcpkg was already bootstrapped** — `vcpkg.exe` was present, just not on PATH. Dependencies are pinned in `vcpkg.json` manifest mode and install into `build/vcpkg_installed/`.

One real gotcha did surface: **nvcc defaults to `-std=c++14`**, silently diverging from the host compiler. `CMAKE_CUDA_STANDARD 17` is set explicitly in `CMakeLists.txt` to prevent that.

---

## 1. The math, and what actually gets implemented

Incompressible Navier–Stokes for a velocity field **u** and pressure *p*:

```
∂u/∂t = -(u·∇)u  -  (1/ρ)∇p  +  ν∇²u  +  f
∇·u = 0
```

Nobody solves that coupled system directly in realtime. The standard move (Stam, *Stable Fluids*, 1999 — the basis of both GPU Gems chapters) is **operator splitting**: advance each term in sequence over one timestep, then restore incompressibility.

Per frame, the pipeline is:

| Step | Operator | Numerical method | Why |
|---|---|---|---|
| 1 | Advection `-(u·∇)u` | Semi-Lagrangian backtrace | Unconditionally stable — no CFL limit. This is *the* trick that makes it realtime. |
| 2 | External force `f` | Explicit Euler | Buoyancy `f = α·T·ŷ − β·ρ·ŷ`, plus mouse/user injection. |
| 3 | Vorticity confinement | Explicit | Re-injects small-scale curl destroyed by diffusive advection. Not in the PDE — a *correction for your own numerical error*. Worth understanding as exactly that. |
| 4 | Diffusion `ν∇²u` | Implicit Jacobi / Gauss–Seidel | For smoke, ν ≈ 0 — often skipped. Implement it, then justify disabling it. |
| 5 | Projection `−∇p` | Solve `∇²p = ∇·u`, then `u ← u − ∇p` | **The hard part.** Helmholtz–Hodge decomposition: any vector field splits into a divergence-free part plus a gradient. Projection discards the gradient. |

**Projection is ~80% of the runtime and 80% of the interesting engineering.** A Poisson solve on a 256³ grid, every frame. Solver progression:

- **Jacobi** — trivial, embarrassingly parallel, slow convergence. Start here.
- **Red–black Gauss–Seidel** — ~2× faster convergence, still parallel. Easy win.
- **Multigrid (V-cycle)** — near-optimal O(n). The genuinely impressive one.

Boundary conditions: no-slip (`u = 0`) on velocity at walls, pure Neumann (`∂p/∂n = 0`) on pressure. Getting these right is where most hobby implementations quietly break — a pure-Neumann Poisson problem is singular, defined only up to a constant, and divergence must sum to zero over the domain or the solver drifts.

---

## 2. Phased build

### Phase 0 — Skeleton (1 weekend)
CMake + vcpkg manifest. GLFW window, OpenGL 4.5 context, Dear ImGui. One CUDA kernel writing a gradient into a `cudaGraphicsGLRegisterImage`-mapped texture, blitted to a fullscreen quad.

> **Exit criterion:** a CUDA kernel's output is on screen at 60 fps with zero CPU readback. This proves the interop path, the only genuinely novel plumbing in the project.

### Phase 1 — 2D solver (2–3 weekends)
Full Stam pipeline at 512², ping-pong buffers, mouse injection of density and velocity. Grayscale density render. Everything in Phase 3 is this code with a `z` index added, so **get the structure right here.**

> **Exit criterion:** interactive 2D smoke, stable for 10+ minutes with no blowup.

### Phase 2 — Validation harness (1–2 weekends) ⭐
**This is the phase that makes the project evidence of understanding rather than a shader toy.** Most people skip it. Don't.

- **Taylor–Green vortex** — an exact analytic solution to Navier–Stokes. Initialize it, step forward, measure L2 error against the closed form.
- **Grid-refinement convergence study** — halve `dx`, verify error drops at the expected order. Semi-Lagrangian advection is 1st-order accurate; this will *show* you that, and motivate the Phase 6 MacCormack/BFECC upgrade with a measured number rather than a vibe.
- **Divergence norm `‖∇·u‖` vs. solver iterations** — plot it. Quantifies how many Jacobi iterations you actually need instead of guessing 40.
- **Mass / energy conservation drift** over time — exposes numerical dissipation.

Dump CSV from C++, plot with Python + matplotlib. These plots are the single most persuasive artifact the project will produce.

### Phase 3 — 3D (2–3 weekends)
3D grids, thermal buoyancy, vorticity confinement. Nothing conceptually new — but memory layout and occupancy start to matter. Debug at 64³ before touching 256³.

### Phase 4 — Volume rendering (3–4 weekends)
Ray-march the density texture from the camera. Then **single-scattering self-shadowing**: a second march toward the light accumulating transmittance. This is the step that takes it from "grey blob" to "that looks like smoke."

GPU Gems 3 ch.30 uses half-angle slicing, a 2007 fixed-function-era optimization. On a 2070 SUPER, **just ray-march it** — simpler code, better quality, plenty fast. Read that chapter for the lighting model, not the slicing scheme.

### Phase 5 — Performance engineering (ongoing)
Nsight Compute. Every kernel here is **memory-bandwidth-bound, not compute-bound** — prove that with a roofline analysis before optimizing anything. Then: red–black GS, multigrid, fp16 storage for density, texture-memory vs. global-memory access patterns, kernel fusion.

### Phase 6 — Stretch
Static obstacles (voxelized boundary conditions) → MacCormack/BFECC advection (fixes the smoke-dissolving-too-fast look) → moving obstacles → combustion/fire model.

---

## 3. Practical vs. not — on *this* hardware

### Comfortably achievable
- 2D at 512²–1024², locked 60 fps, fully interactive.
- **3D at 128³ at 60 fps** — the reliable realtime target.
- **3D at 256³ at ~20–40 fps** — achievable with a tuned solver. Memory: 256³ = 16.8 M cells ≈ 67 MB per float field; with velocity, pressure, divergence, density, temperature and ping-pong copies you land around 0.7–1.2 GB of your 8 GB. Fine. Bandwidth is the real constraint: one Jacobi iteration at 256³ moves ~200 MB, so 40 iterations ≈ 8 GB/frame against ~448 GB/s → **~18 ms in the pressure solve alone.** Which is exactly why the Phase 2 divergence plot and the Phase 5 multigrid work matter.
- Vorticity confinement, buoyancy, single-scattering volume rendering, static obstacles.

### Possible but a real fight
- **Multigrid Poisson solver** — big payoff, genuinely fiddly (restriction/prolongation operators, boundary handling at every level). Budget several weekends. Very impressive when it lands.
- **512³ realtime** — ~537 MB *per field*, ~6–7 GB total. Fits, barely, but bandwidth puts you in single-digit fps. Viable for offline frame dumps, not interaction.
- **FLIP/PIC hybrid** — particle-grid transfer, sorting, different data structures. That's a second project, not a feature.

### Not worth attempting solo
- Sparse / adaptive grids (NanoVDB, OpenVDB) — enormous scope, only pays off past 1024³.
- Full multiple-scattering or path-traced smoke at interactive rates.
- Production-grade combustion (Houdini-class) — years of person-effort.

---

## 4. Risks, ordered by how likely they are to bite

1. **CUDA–GL interop plumbing** (Phase 0) — most likely place to lose a weekend to a black screen. Isolate it before any physics exists.
2. **Boundary conditions** — the classic silent killer. Symptoms: smoke sticking to walls, pressure drifting, slow mass leaks. The Phase 2 divergence plot catches this.
3. **CMake + CUDA 13 + MSVC** — upgrade CMake first and this mostly evaporates.
4. **Debugging on the GPU** — no practical `printf` breakpoint workflow. Mitigation: keep a CPU reference implementation of the 2D solver and diff fields against it. Slow, but it will find the bug, and it's worth the afternoon it costs to write.
5. **Scope creep toward rendering.** Volume rendering is fun and endless. The stated goal is math→application, so protect Phase 2.

---

## 5. Proposed repo layout

```
Fluid_Smoke/
├── CMakeLists.txt
├── vcpkg.json                  # glfw3, glad, glm, imgui
├── src/
│   ├── main.cpp                # window, loop, ImGui controls
│   ├── gl/                     # shaders, interop, fullscreen quad, raymarcher
│   ├── sim/
│   │   ├── Solver2D.{h,cu}
│   │   ├── Solver3D.{h,cu}
│   │   ├── kernels/            # advect.cu, project.cu, vorticity.cu, boundary.cu
│   │   └── Field.h             # ping-pong buffer abstraction — get this right early
│   └── validate/               # Taylor-Green, convergence study, CSV dump
├── tools/plots/                # Python: matplotlib from the CSVs
└── docs/
    ├── derivation.md           # the splitting scheme, written out in your own words
    └── results/                # convergence plots, Nsight captures, screenshots
```

`docs/derivation.md` is not decoration — writing the derivation out longhand is how you find the parts you don't actually understand yet.

---

## 6. Reading order

1. **Stam 1999, *Stable Fluids*** — read this *first*, before either GPU Gems chapter. It's the source; both chapters are GPU ports of it. ~8 pages.
2. **GPU Gems 1, ch.38** — the 2D algorithm, clearest pseudocode. Mentally translate its fragment-shader ping-ponging into CUDA kernels plus double-buffered `cudaSurfaceObject_t`.
3. **GPU Gems 3, ch.30** — 3D, buoyancy, vorticity confinement, obstacles, volume rendering. Take the physics and the lighting model; skip the half-angle slicing.
4. **Bridson, *Fluid Simulation for Computer Graphics*** — the reference when a boundary condition or the pressure solve stops making sense. Worth owning.

---

## 7. First three concrete actions

1. Upgrade CMake to 3.28+.
2. Bootstrap vcpkg (`cd C:\vcpkg && .\bootstrap-vcpkg.bat`), add it to PATH.
3. Read Stam 1999 end to end before writing a line of code.

Then Phase 0: get one CUDA kernel's output onto the screen.
