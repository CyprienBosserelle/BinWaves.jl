"""
    BinWaves

BinWaves-style additive hybrid downscaling with the WaveAction / WaveSweep model.

Method (Cagigal, Méndez, Ricondo, Gutiérrez-Barceló, Bosserelle, Hoeke 2024, Ocean Modelling 189, 102346 —
"BinWaves: An additive hybrid method to downscale directional wave spectra to nearshore areas"; reference implementation:
GeoOcean/BlueMath_tk `bluemath_tk.waves.binwaves`):

  1. Split the (frequency, direction) plane into bins. Each bin is one *case*: a narrow-banded wave train carrying only
     that bin's energy (Hs = 1 m, 0.1 m for periods < 5 s for numerical stability), imposed on the open boundaries.
  2. Propagate every case once to the nearshore with the spectral wave model (here: stationary four-sweep solver).
     The result is a set of transfer coefficients KP: the fraction of the case's input variance that ends up in each
     output bin (or, integrated, the output variance m0) at each location. This is the expensive step, done once.
  3. For any offshore spectrum E(t, f, θ), the variance in each bin, V_c(t), weights the cases and
         E_nearshore(t, f, θ, x) = Σ_c V_c(t) · KP_c(f, θ, x)          (linear superposition)
     which needs no wave-model run at all: a matrix product per time series.

Differences from the paper's set-up (all by design, to suit this model):
  * The case bins are the model's own spectral bins (optionally merged in groups: `fgroup`, `dgroup`), so a case is an exact
    bin-pure boundary spectrum (explicit-matrix `SpectrumBC`) instead of a JONSWAP γ = 50 / small-spread approximation of one.
  * Full nearshore spectra are stored only at the `points` you ask for; the gridded fields store the integrated fraction of
    input variance (so Hs² = 16 Σ V_c KP_c, exactly the paper's 'Hs of the unit case = Kp').
  * Linearity: exact for refraction, shoaling and the (linear) JONSWAP bottom friction; approximate when depth-induced
    breaking is on (non-linear in wave height) — as in the paper, where cases use Hs = 1 m — and when wind input is on.

    using BinWaves                 # brings WaveAction, WaveSweep, WaveBlocks and WaveGPU exports into scope

    kp = build_kp(model; points = [(40, 20)], sides = (:left, :bottom))        # slow part, once
    rec = reconstruct(kp, E, f_hz, dir_nautical_from_deg)                       # fast part, per offshore time series
    rec.Hs                 # (nx, ny, nt) nearshore Hs

    # long builds: checkpoint to NetCDF, resume after an interruption, reuse later without rebuilding
    kp = build_kp(model; points = pts, checkpoint = "kp.nc")       # re-running the same call resumes from kp.nc
    kp = load_kp_netcdf("kp.nc")                                   # in another session; then reconstruct(kp, E, f, dir)
    save_kp_netcdf("copy.nc", kp)                                  # write any BinWavesKP
    point_parameters(kp, rec.point_var)                                        # Hs, Tm01, Tp, Dir at the points
"""
module BinWaves

using Printf, Serialization, LinearAlgebra
using NetCDF

# Core solvers (submodules)
include("WaveAction.jl")   # explicit spectral wave action solver
include("WaveSweep.jl")    # implicit / stationary four-sweep solver
include("WaveBlocks.jl")   # block-adaptive (quadtree) meshes for both solvers
include("WaveGPU.jl")      # KernelAbstractions (GPU / portable CPU) backend

using .WaveAction
using .WaveSweep
using .WaveBlocks
using .WaveGPU

# Submodules, for qualified access to internals (e.g. `WaveAction.NG`)
export WaveAction, WaveSweep, WaveBlocks, WaveGPU

# Re-export the public API of the submodules
export CartesianGrid, SpectralGrid, WaveParams, WaveModel, SpectrumBC, OpenBC, ZeroBC, ExternalBC
export set_boundaries!, initialize!, set_environment!, update_environment!, set_wind!
export GSEParams, set_gse!, gse_step!
export step!, run!, stable_dt, bulk, total_action, jonswap, wavenumber, wavekin
export naut2math, math2naut

export SweepParams, four_sweeps!, step_implicit!, run_implicit!, solve_stationary!, hs_tm

export BlockMesh, refine!, refine_by_depth!, balance!, nblocks, ncells, block_bounds, block_dx, cell_centres, cell_dx,
       cell_index, locate, grid_index, to_grid, bilinear, BlockModel, cell_depth, cell_wet, mesh_summary

export GPUModel, gpu_model, to_cpu!, upload_state!, refresh_environment!, refresh_boundaries!,
       step_gpu!, run_gpu!, step_implicit_gpu!, run_implicit_gpu!, solve_stationary_gpu!, bulk_gpu, hs_tm_gpu, total_action_gpu

# BinWaves downscaling (transfer coefficients)
include("kp.jl")

export BinWavesKP, build_kp, reconstruct, point_parameters, save_kp, load_kp, bin_offshore_spectrum,
       save_kp_netcdf, load_kp_netcdf

# Local wind sea (non-linear in wind speed: look-up table over wind speed and direction)
include("windkp.jl")

export WindKP, build_wind_kp, wind_sea, save_wind_kp_netcdf, load_wind_kp_netcdf

end # module BinWaves
