# BinWaves.jl

Spectral wave action models on a Cartesian grid (explicit, implicit four-sweep and stationary solvers, CPU threads or GPU
through KernelAbstractions) and BinWaves additive hybrid downscaling of directional wave spectra to nearshore areas
(Cagigal et al. 2024, Ocean Modelling 189, 102346).

## Layout
```
Project.toml        package metadata and dependencies
src/BinWaves.jl     package module: includes the submodules below and re-exports their API
src/WaveAction.jl   explicit wave action balance solver (submodule WaveAction)
src/WaveSweep.jl    implicit / stationary four-sweep Gauss–Seidel solver (submodule WaveSweep)
src/WaveGPU.jl      KernelAbstractions GPU / portable backend (submodule WaveGPU)
src/kp.jl           BinWaves transfer coefficients: build_kp, reconstruct, NetCDF checkpointing
src/windkp.jl       local wind-sea look-up table: build_wind_kp, wind_sea (added to reconstruct)
test/               package tests (Pkg.test())
examples/           garden sprinkler effect test cases, SWAN comparison input
benchmark/          micro-benchmarks of the hot paths
docs/               method notes and figures
```

## Install / use
```julia
using Pkg
Pkg.develop(path = "path/to/BinWaves.jl")     # or Pkg.add(url = "https://github.com/CyprienBosserelle/BinWaves.jl")

using BinWaves
grid  = CartesianGrid(x, y)
model = WaveModel(grid; depth = D)
set_boundaries!(model; left = SpectrumBC(Hs = 2.0, Tp = 10.0, dir = 270.0))
solve_stationary!(model)

kp  = build_kp(model; points = [(40, 20)], sides = (:left,))   # slow part, once
rec = reconstruct(kp, E, f_hz, dir_nautical_from_deg)          # fast part, per offshore time series
```
Local wind sea (semi-enclosed water, see `docs/README_Wind.md`):
```julia
wk  = build_wind_kp(model; dirs = 0:22.5:337.5, speeds = [5, 10, 15, 20, 25, 30], points = [(40, 20)])   # once
rec = reconstruct(kp, E, f_hz, dir_nautical_from_deg; wind = wk, U10 = u10, wdir = wind_dir_from)       # Hs² = swell + wind sea
```

`using BinWaves` exports the public API of all submodules; internals are reachable as `WaveAction.NG`, etc.
For a GPU, also load the backend package (e.g. `using CUDA`) and pass `backend = CUDABackend()` to `gpu_model`.

## Tests, examples, benchmarks
```
julia -t auto --project=. -e 'using Pkg; Pkg.test()'
julia -t auto --project=examples examples/gse_tolman_test.jl
julia -t 1    --project=benchmark benchmark/bench.jl
```
The `examples` and `benchmark` environments point to the package through `[sources]` (Julia ≥ 1.11); on older Julia run
`Pkg.develop(path = "..")` in them once.

See `docs/README_WaveAction.md`, `docs/README_GSE.md` and `docs/README_Wind.md` for the solver and garden-sprinkler-effect notes.
