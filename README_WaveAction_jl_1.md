# WaveAction.jl – multi-threaded CPU wave action balance solver (Cartesian grid)

Julia port of the BG_Wave GPU engine for local testing. Single file, **no package dependencies** (only `Base.Threads`,
plus `Test`/`Printf` for the tests). Tested with Julia 1.13.3 (should work on ≥ 1.9).

## Run
```
julia -t auto test_wave.jl               # 20 checks, ~1 min
julia -t auto example_shoal_beach.jl     # shoal + beach demo
```
`-t auto` (or `-t 8`) sets the number of threads. Parallelism is shared-memory threading over grid rows; it is not
`Distributed`/MPI.

## Use
```julia
include("WaveAction.jl"); using .WaveAction
grid  = CartesianGrid(x, y)                      # cell-centre coordinates, uniform (dx may differ from dy)
spec  = SpectralGrid(ns = 24, nth = 36, fmin = 0.05, fmax = 0.6)
model = WaveModel(grid; depth = D, spec = spec,  # D (nx,ny) positive down; or zb = ..., zs = 0.0
                  params = WaveParams(order = 2, breaking = true, friction = true, hmin = 0.2),
                  U = U, V = V)                  # optional currents (nx,ny)
set_boundaries!(model; left = SpectrumBC(Hs = 2.0, Tp = 10.0, dir = 270.0))   # others default OpenBC()
run!(model; tend = 3600.0, output_dt = 600.0, callback = (m, t) -> println(t))
out = bulk(model)                                # Hs, Tm, Dir, Sxx, Sxy, Syy  (nx,ny) matrices
```
* Directions in/out are nautical "coming from" degrees (270 = from the west, travelling towards +x).
* Boundaries: `OpenBC()` (default, zero-gradient: outgoing waves leave, any incoming ones are extrapolated from the interior), `ZeroBC()` (no incoming waves), `SpectrumBC(...)`.
* `SpectrumBC(N)` imposes an explicit (ns,nth) action spectrum; `scale = t -> ...` scales Hs in time.
* `set_environment!(model; depth, U, V, dddt)` updates depth/currents (tide, surge, a hydrodynamic model) and refreshes
  the derived fields and the time step. `set_wind!` sets a uniform or per-cell wind (enable with `wind = true`).
* `initialize!(model; Hs, Tp, dir)` sets a uniform initial sea state. `total_action(model)` is a conservation diagnostic.
* Garden sprinkler effect: `set_gse!(model; alpha_s = 3.0, alpha_n = 3.0)` (see `README_GSE.md`).
* `WaveParams` switches: `refraction`, `wind`, `whitecap`, `breaking`, `friction`, `order` (1/2), `theta` (limiter 1–2).

## Differences from the CUDA version
Same equations, scheme (limited MUSCL upwind, midpoint predictor/corrector, CFL ≤ 0.5) and source terms, but: plain
uniform grid with 2 ghost layers (no quadtree/halo machinery); kinematics (face group velocities, refraction and
σ-shift coefficients, time step) are precomputed once per environment update so the time loop never solves the dispersion
relation; the σ-shift part is skipped when there are no currents and no `dddt`; each cell evaluates its own face fluxes
(identical on both sides, so conservative to round-off and race-free).

## What was verified (Julia 1.13.3, here run on a single core)
* Dispersion accuracy (1e-12), exact discrete Hs of the JONSWAP spectrum, deep-water radiation stress.
* Green's-law shoaling with refraction off: N·cg constant per bin to 0.55 % (order 1) and 0.05 % (order 2).
* Snell's law on a plane beach: mean direction within 0.5° of the prediction (18.2/15.6/12.4° vs 18.1/15.3/12.0°).
* Conservation of total wave action ≈ 4e-6 (only top-of-spectrum outflow) and non-negativity with depth gradients and
  currents; first-order loses action only when its diffusion tail reaches an open boundary.
* Source terms: breaking saturates Hrms below γd and is stable for dt = 1e4 s; friction decay; wind + whitecapping stay
  bounded (Hs 3.3 m after 12 h at 15 m/s, fully developed ≈ 5.5 m).
* Results are **bit-identical with 1 and 4 threads**. Parallel speed-up was **not measured** (single-core sandbox).

## Limitations
* No nonlinear quadruplet/triad interactions (wind-sea spectral shape is not realistic); no wave reflection; `OpenBC` (zero-gradient: incoming waves are extrapolated from the interior), `ZeroBC` (no incoming waves) and imposed-spectrum
  boundaries only; Cartesian only; Float64 only.
* Whitecapping/wind/limiter constants follow the author's recollection of SWAN's Komen set; check them against the SWAN
  technical documentation before scientific use.
* Explicit θ/σ advection: strong refraction near the shoreline can limit dt (raise `hmin`, smooth the bathymetry).
* Memory ≈ 3 × ns·nth·(nx+4)(ny+4) × 8 bytes (e.g. 500×500 cells, 24×36 bins ≈ 5 GB).

## Performance (v2)
Per-environment quantities are now computed once in `update_environment!` rather than every step: wave number and
1/sinh²(kd) per cell and frequency, and the refraction coefficients (depth/current gradient terms) per cell and direction.
Cells where refraction and current shift are exactly zero (deep water, no currents) skip the θ and σ terms entirely. The explicit tendency
computes each face flux once (not once from each side), uses reciprocal tables for the frequency-axis limiter, and
vectorises the frequency loops. In the implicit solver the refraction coefficients, dissipation rates (refreshed once per sweep),
and the per-cell source setup are no longer recomputed in every cell visit, and the cyclic tridiagonal solve is replaced by a
plain tridiagonal solve over each run of active directions (the cyclic solve is only used if all directions are active).
`julia -t 1 bench.jl` times the hot paths. Single core, 100×60 cells, 16×24 bins:

| case | explicit step before → after | 4 sweeps before → after |
|---|---|---|
| variable depth | 173 → 82 ms | 162 → 94 ms |
| flat deep water | 140 → 46 ms | 189 → 82 ms |
| depth + currents | 288 → 106 ms | 181 → 89 ms |
| depth + all sources | 234 → 91 ms | 260 → 100 ms |

Results are unchanged to round-off (explicit and implicit, ~1e-16 relative); the stationary solution with sources differs by 1e-8.
