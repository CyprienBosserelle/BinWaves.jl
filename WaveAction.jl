"""
    WaveAction

Spectral wave action balance solver on a uniform Cartesian grid (CPU, multi-threaded, no dependencies).

Julia port of the BG_Wave GPU engine (same physics and numerics, but a plain Cartesian grid instead of the
BG_Flood block-uniform quadtree):

    dN/dt + d(cx N)/dx + d(cy N)/dy + d(cσ N)/dσ + d(cθ N)/dθ = S(σ,θ)/σ ,   N = E/σ

* explicit finite volume, limited MUSCL upwind (generalised minmod), order 1 or 2
* midpoint predictor/corrector time stepping (the FlowGPU structure), CFL ≤ 0.5
* sources (wind, whitecapping, depth-induced breaking, bottom friction) applied semi-implicitly after transport
* refraction by depth and currents, frequency shift by currents / changing depth
* threading over grid rows with `Threads.@threads` (start Julia with `julia -t auto`)

Conventions
* x east, y north; depth `d` positive down (m); currents U (x), V (y) in m/s
* Directions given to / returned by the API are NAUTICAL "coming from", degrees clockwise from North
  (waves from the west = 270). Internally θ is the mathematical "going to" angle (rad, from +x, counter-clockwise).
* Spectral axes: ns log-spaced frequencies × nth directions covering the full circle. Per-cell spectrum is stored
  contiguously: `N[is, it, i, j]`, with 2 ghost cells on every side of the (nx, ny) domain.

Quick start
```julia
include("WaveAction.jl"); using .WaveAction
grid  = CartesianGrid(x, y)                       # cell-centre coordinates (uniform)
model = WaveModel(grid; depth = D)                # D :: Matrix (nx, ny), positive down
set_boundaries!(model; left = SpectrumBC(Hs = 2.0, Tp = 10.0, dir = 270.0))
run!(model; tend = 1800.0)
out = bulk(model)                                 # out.Hs, out.Tm, out.Dir, out.Sxx, out.Sxy, out.Syy
```
"""
module WaveAction

using Base.Threads

export CartesianGrid, SpectralGrid, WaveParams, WaveModel, SpectrumBC, OpenBC, ZeroBC
export set_boundaries!, initialize!, set_environment!, update_environment!, set_wind!
export GSEParams, set_gse!, gse_step!
export step!, run!, stable_dt, bulk, total_action, jonswap, wavenumber, wavekin
export naut2math, math2naut

const NG = 2   # ghost layers

# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------
"nautical 'coming from' degrees -> mathematical 'going to' radians"
naut2math(deg) = (270.0 - deg) * π / 180
"mathematical 'going to' radians -> nautical 'coming from' degrees in [0,360)"
math2naut(rad) = mod(270.0 - rad * 180 / π, 360.0)

@inline function minmod3(a, b, c)
    # branch-free (identical value to the usual if/elseif form): vectorisable
    mn = ifelse(b < a, b, a); mn = ifelse(c < mn, c, mn)
    mx = ifelse(b > a, b, a); mx = ifelse(c > mx, c, mx)
    return ifelse(mn > 0, mn, ifelse(mx < 0, mx, zero(a)))
end
"limited slope per cell index (uniform spacing), generalised minmod with θ in [1,2]"
@inline limslope(Nm, N0, Np, θ) = minmod3(θ * (N0 - Nm), 0.5 * (Np - Nm), θ * (Np - N0))
"limited slope dN/dx on a non-uniform axis (frequency axis)"
@inline limslope_nu(Nm, N0, Np, xm, x0, xp, θ) =
    minmod3(θ * (N0 - Nm) / (x0 - xm), (Np - Nm) / (xp - xm), θ * (Np - N0) / (xp - x0))

# ------------------------------------------------------------------------------------------------
# Dispersion and propagation speeds
# ------------------------------------------------------------------------------------------------
"""
    wavenumber(σ, d, g=9.81)

Solve σ² = g k tanh(k d) for k (Hunt 1979 first guess + two Newton iterations).
"""
@inline function wavenumber(σ, d, g = 9.81)
    y = σ * σ * d / g
    y2 = y * y
    kd = sqrt(y2 + y / (1 + 0.6522y + 0.4622y2 + 0.0864y2 * y2 + 0.0675y2 * y2 * y))
    for _ in 1:2
        t = tanh(kd)
        kd -= (kd * t - y) / (t + kd * (1 - t * t))
    end
    return kd / d
end

"""
    wavekin(σ, d, g=9.81) -> (k, cg, dσdd)

Wave number, group velocity and ∂σ/∂d.
"""
@inline function wavekin(σ, d, g = 9.81)
    k = wavenumber(σ, d, g)
    kd2 = 2 * k * d
    if kd2 > 40.0
        return k, 0.5 * σ / k, 0.0
    end
    s2 = sinh(kd2)
    return k, 0.5 * σ / k * (1 + kd2 / s2), σ * k / s2
end

# ------------------------------------------------------------------------------------------------
# Types
# ------------------------------------------------------------------------------------------------
"""
    CartesianGrid(x, y)

Uniform grid from cell-centre coordinate vectors. `dx`, `dy` may differ.
"""
struct CartesianGrid
    x::Vector{Float64}
    y::Vector{Float64}
    nx::Int
    ny::Int
    dx::Float64
    dy::Float64
    function CartesianGrid(x::AbstractVector, y::AbstractVector)
        length(x) >= 1 && length(y) >= 1 || error("grid needs at least one cell in each direction")
        dx = length(x) > 1 ? Float64(x[2] - x[1]) : 1.0
        dy = length(y) > 1 ? Float64(y[2] - y[1]) : 1.0
        (all(isapprox.(diff(x), dx; rtol = 1e-6)) || length(x) == 1) || error("x must be uniformly spaced")
        (all(isapprox.(diff(y), dy; rtol = 1e-6)) || length(y) == 1) || error("y must be uniformly spaced")
        dx > 0 && dy > 0 || error("x and y must be increasing")
        new(Float64.(x), Float64.(y), length(x), length(y), dx, dy)
    end
end

"""
    SpectralGrid(; ns=24, nth=36, fmin=0.05, fmax=0.6)

Logarithmically spaced frequencies (Hz) and uniform directions over the full circle.
"""
struct SpectralGrid
    ns::Int
    nth::Int
    fmin::Float64
    fmax::Float64
    fac::Float64            # σ(k+1)/σ(k)
    rf::Float64             # sqrt(fac)
    dth::Float64
    sig::Vector{Float64}    # ns
    sigf::Vector{Float64}   # ns+1 faces
    dsig::Vector{Float64}   # ns
    cth::Vector{Float64}
    sth::Vector{Float64}
    ch::Float64             # cos(dth/2)
    sh::Float64             # sin(dth/2)
    rdp::Vector{Float64}    # rdp[k] = 1/(σ[k+1]-σ[k])   (0 for k = ns)
    rdc::Vector{Float64}    # rdc[k] = 1/(σ[k+1]-σ[k-1]) (0 at the ends)
end
function SpectralGrid(; ns::Int = 24, nth::Int = 36, fmin::Real = 0.05, fmax::Real = 0.6)
    ns >= 3 || error("ns must be >= 3")
    nth >= 4 || error("nth must be >= 4")
    0 < fmin < fmax || error("need 0 < fmin < fmax")
    fac = (fmax / fmin)^(1 / (ns - 1))
    rf = sqrt(fac)
    dth = 2π / nth
    sig = [2π * fmin * fac^(k - 1) for k in 1:ns]
    sigf = [sig[1] / rf; sig .* rf]
    dsig = sig .* (rf - 1 / rf)
    cth = [cos((it - 1) * dth) for it in 1:nth]
    sth = [sin((it - 1) * dth) for it in 1:nth]
    rdp = [k < ns ? 1 / (sig[k+1] - sig[k]) : 0.0 for k in 1:ns]
    rdc = [(k > 1 && k < ns) ? 1 / (sig[k+1] - sig[k-1]) : 0.0 for k in 1:ns]
    SpectralGrid(ns, nth, Float64(fmin), Float64(fmax), fac, rf, dth, sig, sigf, dsig, cth, sth, cos(dth / 2), sin(dth / 2), rdp, rdc)
end

"""
    WaveParams(; kwargs...)

Physics and numerics switches. Wind direction `Wdir` is nautical 'coming from' (degrees).
Constants of whitecapping / wind input follow the SWAN Komen set as remembered by the author: check against the
SWAN technical documentation before scientific use.
"""
Base.@kwdef struct WaveParams
    g::Float64 = 9.81
    order::Int = 2                  # 1 first order, 2 limited MUSCL
    theta::Float64 = 1.3            # limiter parameter in [1,2] (1 = minmod)
    hmin::Float64 = 0.05            # wave 'dry' threshold (m)
    refraction::Bool = true         # θ advection and σ shift
    wind::Bool = false
    U10::Float64 = 0.0
    Wdir::Float64 = 270.0           # nautical from (deg)
    rhoratio::Float64 = 1.2 / 1025
    whitecap::Bool = false
    Cds::Float64 = 2.36e-5
    delta_ds::Float64 = 0.0
    p_ds::Float64 = 2.0
    stpm::Float64 = 3.02e-3
    breaking::Bool = false
    alpha_bj::Float64 = 1.0
    gamma_bj::Float64 = 0.73
    friction::Bool = false
    Cbf::Float64 = 0.038
    Cl::Float64 = 0.1               # wind growth limiter
    alpha_pm::Float64 = 0.0081
    rhog::Float64 = 1025.0 * 9.81   # radiation stress scaling (N/m3)
end

abstract type BoundaryCondition end
"""
    OpenBC()

Zero-gradient boundary: ghost cells copy the nearest interior cell. Outgoing waves leave freely; any
*incoming* waves are extrapolated from the interior (this is what makes a y-uniform problem work with open
side boundaries). Use `ZeroBC()` if no energy must enter.
"""
struct OpenBC <: BoundaryCondition end
"""
    ZeroBC()

No incoming waves (ghost cells hold zero action); outgoing waves leave freely. The SWAN default at open boundaries.
"""
struct ZeroBC <: BoundaryCondition end
"""
    SpectrumBC(; Hs, Tp, dir, gamma=3.3, spread=10.0, scale=t->1.0)
    SpectrumBC(N::Matrix; scale=t->1.0)

Impose a spectrum on a boundary. Parametric JONSWAP × cos^{2s} (dir nautical-from, degrees), or an explicit
(ns, nth) wave action matrix. `scale(t)` multiplies Hs in time (N is scaled by scale²).
"""
struct SpectrumBC <: BoundaryCondition
    Hs::Float64
    Tp::Float64
    dir::Float64
    gamma::Float64
    spread::Float64
    N::Union{Nothing,Matrix{Float64}}
    scale::Function
end
SpectrumBC(; Hs, Tp = 10.0, dir = 270.0, gamma = 3.3, spread = 10.0, scale = t -> 1.0) =
    SpectrumBC(Float64(Hs), Float64(Tp), Float64(dir), Float64(gamma), Float64(spread), nothing, scale)
SpectrumBC(N::AbstractMatrix; scale = t -> 1.0) = SpectrumBC(0.0, 10.0, 270.0, 3.3, 10.0, Matrix{Float64}(N), scale)

mutable struct Environment
    d::Matrix{Float64}; U::Matrix{Float64}; V::Matrix{Float64}; dddt::Matrix{Float64}
    dddx::Matrix{Float64}; dddy::Matrix{Float64}
    dUdx::Matrix{Float64}; dUdy::Matrix{Float64}; dVdx::Matrix{Float64}; dVdy::Matrix{Float64}
    wet::Matrix{Bool}
    cgx::Array{Float64,3}; cgy::Array{Float64,3}        # (ns, nx+4, ny+4) group velocity at left / bottom faces
    Ufx::Matrix{Float64}; Vfy::Matrix{Float64}          # face currents
    wetx::Matrix{Bool}; wety::Matrix{Bool}
    ssh::Array{Float64,3}                               # σ/sinh(2kd) per cell and bin
    csig::Array{Float64,4}                              # (4, ns+1, nx+4, ny+4): cσ = A + B c² + C c s + D s²
    dtcell::Matrix{Float64}
    u10x::Matrix{Float64}; u10y::Matrix{Float64}
    has_shift::Bool
    # ---- derived once per environment update (depth / currents do not change during a run) ----
    kk::Array{Float64,3}                                # (ns, ·, ·) wave number
    sh2i::Array{Float64,3}                              # (ns, ·, ·) 1/sinh²(min(kd,30))  (bottom friction)
    gdp::Array{Float64,3}; gdm::Array{Float64,3}        # (nth, ·, ·) θ-refraction: c_θ = ssh·gd + gU at the +/- θ faces
    gUp::Array{Float64,3}; gUm::Array{Float64,3}
    refr::Matrix{Bool}                                  # cell has non-zero refraction / current shift (else θ,σ terms skipped)
    Fx::Array{Float64,3}; Fy::Array{Float64,3}          # (ns, ·, ·) scratch face fluxes of one direction
    dec::Array{Float64,3}                               # (ns, ·, ·) dissipation rate (implicit solver, refreshed each iteration)
end

"""
    GSEParams(; alpha_s = 3.0, alpha_n = 3.0, cfl = 1.0, tau = 0.0, maxsub = 400)

Garden-sprinkler-effect (GSE) alleviation by anisotropic diffusion of every spectral bin in physical space
(Booij & Holthuijsen 1987), with the coefficients of Tolman (2002) cast in diffusion form by Popinet et al. (2010):

    ∂N/∂τ = ∇·(D ∇N),   D = Dss s sᵀ + Dnn n nᵀ   (s = propagation direction, n = normal)
    Dss = alpha_s² (Δcg)² Ts / 3 ,   Δcg = (γ - 1/γ) cg / 2     (γ = frequency ratio of the spectral grid)
    Dnn = alpha_n² (cg Δθ)² Ts / 3

applied over each solver step (operator splitting, after transport and sources).

`Ts` is the *propagation time step of the frequency component*. In WAVEWATCH III and Gerris every frequency is advanced
with its own step of about Δx/cg, so that the filter displaces the action by a fixed fraction of the grid size
(≈ alpha_s (γ-1/γ)/2 · Δx and alpha_n Δθ · Δx). To reproduce that with a single global solver step, Ts is set per
face from the local group velocity:

* `cfl > 0` (default 1.0): `Ts = cfl · min(Δx,Δy) / cg`  → smoothing independent of the solver's time step
  (cfl = 1 is the Gerris choice; D then scales with cg).
* else `tau > 0`: a fixed filter time scale (s) for all frequencies.
* else: `Ts` = the solver step length (WW3 with one global step; weak for the slow components and dependent on dt).

`alpha_s`, `alpha_n`: strength along / across the propagation direction. **Popinet et al. found that their diffusion
form needs values twice those of WAVEWATCH III's averaging scheme for the same result: WW3's customary 1.5 corresponds
to 3.0 here.** Stability of the filter requires (alpha·offset)² < 3Δx², which the default parameters satisfy; `maxsub` caps the
number of explicit diffusion sub-steps per solver step (needed for large implicit steps).
"""
Base.@kwdef struct GSEParams
    alpha_s::Float64 = 3.0
    alpha_n::Float64 = 3.0
    cfl::Float64 = 1.0
    tau::Float64 = 0.0
    maxsub::Int = 400
    safety::Float64 = 2.5      # explicit stability margin: nsub = ceil(safety · dt · (Dss+Dnn)max / min(Δx,Δy)²)  (limit is 2)
    adi_above::Int = 2         # use the unconditionally stable ADI step when the explicit scheme would need more sub-steps than this
end

mutable struct WaveModel
    grid::CartesianGrid
    spec::SpectralGrid
    params::WaveParams
    N::Array{Float64,4}      # (ns, nth, nx+4, ny+4)
    N1::Array{Float64,4}
    dN::Array{Float64,4}
    env::Environment
    bc::NTuple{4,BoundaryCondition}   # left, right, bottom, top
    Nbc::Vector{Matrix{Float64}}      # cached boundary spectra
    t::Float64
    gse::Union{Nothing,GSEParams}     # garden sprinkler effect alleviation (nothing = off)
    gsec::Any                         # cache of face diffusivities (rebuilt when the environment or GSE settings change)
end

# ------------------------------------------------------------------------------------------------
# JONSWAP spectrum on the model grid (discrete Hs exactly as requested)
# ------------------------------------------------------------------------------------------------
"""
    jonswap(spec, Hs, Tp, dir; gamma=3.3, spread=10.0) -> Matrix (ns, nth)

Wave action N(σ,θ) of a JONSWAP × cos^{2s}((θ-θ0)/2) spectrum, normalised so that the discrete
Hs = 4 sqrt(m0) equals `Hs`. `dir` is nautical-from degrees.
"""
function jonswap(spec::SpectralGrid, Hs, Tp, dir; gamma = 3.3, spread = 10.0)
    N = zeros(spec.ns, spec.nth)
    Hs <= 0 && return N
    fp = 1 / Tp
    th0 = naut2math(dir)
    Ef = map(spec.sig) do σ
        f = σ / 2π
        sg = f <= fp ? 0.07 : 0.09
        f^-5 * exp(-1.25 * (fp / f)^4) * gamma^exp(-(f - fp)^2 / (2sg^2 * fp^2))
    end
    D = map(1:spec.nth) do it
        dd = (it - 1) * spec.dth - th0
        dd = atan(sin(dd), cos(dd))
        max(cos(dd / 2), 0.0)^(2spread)
    end
    m0 = sum(Ef[is] * D[it] * spec.dsig[is] * spec.dth for is in 1:spec.ns, it in 1:spec.nth)
    A = (Hs / 4)^2 / m0
    for it in 1:spec.nth, is in 1:spec.ns
        N[is, it] = A * Ef[is] * D[it] / spec.sig[is]
    end
    return N
end

# ------------------------------------------------------------------------------------------------
# Model construction
# ------------------------------------------------------------------------------------------------
function fill_ghost!(A::AbstractMatrix)
    nxg, nyg = size(A)
    for j in 1:nyg, g in 1:NG
        A[g, j] = A[NG+1, j]
        A[nxg-g+1, j] = A[nxg-NG, j]
    end
    for i in 1:nxg, g in 1:NG
        A[i, g] = A[i, NG+1]
        A[i, nyg-g+1] = A[i, nyg-NG]
    end
    return A
end

function pad(A::AbstractMatrix, nx, ny)
    size(A) == (nx, ny) || error("array must have size ($nx, $ny), got $(size(A))")
    P = zeros(nx + 2NG, ny + 2NG)
    P[NG+1:NG+nx, NG+1:NG+ny] .= A
    return fill_ghost!(P)
end

"""
    WaveModel(grid; depth=nothing, zb=nothing, zs=0.0, U=nothing, V=nothing, spec=SpectralGrid(),
              params=WaveParams(), Hs0=0.0, Tp0=10.0, dir0=270.0)

Create a wave model. Give either `depth` (positive down, (nx,ny) matrix) or the bed level `zb` (and still water
level `zs`). Optional currents `U`, `V` (m/s). The initial sea state is calm unless `Hs0 > 0`.
"""
function WaveModel(grid::CartesianGrid; depth = nothing, zb = nothing, zs = 0.0, U = nothing, V = nothing,
                   spec::SpectralGrid = SpectralGrid(), params::WaveParams = WaveParams(),
                   Hs0 = 0.0, Tp0 = 10.0, dir0 = 270.0, gamma0 = 3.3, spread0 = 10.0)
    nx, ny = grid.nx, grid.ny
    (depth === nothing) == (zb === nothing) && error("give exactly one of `depth` or `zb`")
    D = depth === nothing ? max.(zs .- zb, 0.0) : Float64.(depth)
    ns, nth = spec.ns, spec.nth
    ng = (nx + 2NG, ny + 2NG)
    z2() = zeros(ng...)
    env = Environment(pad(D, nx, ny),
        U === nothing ? z2() : pad(U, nx, ny),
        V === nothing ? z2() : pad(V, nx, ny),
        z2(), z2(), z2(), z2(), z2(), z2(), z2(),
        fill(false, ng...),
        zeros(ns, ng...), zeros(ns, ng...), z2(), z2(), fill(false, ng...), fill(false, ng...),
        zeros(ns, ng...), zeros(4, ns + 1, ng...), fill(Inf, ng...), z2(), z2(), false,
        zeros(ns, ng...), zeros(ns, ng...), zeros(nth, ng...), zeros(nth, ng...), zeros(nth, ng...), zeros(nth, ng...),
        fill(false, ng...), zeros(ns, ng...), zeros(ns, ng...), zeros(ns, ng...))
    m = WaveModel(grid, spec, params,
        zeros(ns, nth, ng...), zeros(ns, nth, ng...), zeros(ns, nth, ng...), env,
        (OpenBC(), OpenBC(), OpenBC(), OpenBC()), [zeros(ns, nth) for _ in 1:4], 0.0, nothing, nothing)
    set_wind!(m; U10 = params.U10, dir = params.Wdir)
    update_environment!(m)
    initialize!(m; Hs = Hs0, Tp = Tp0, dir = dir0, gamma = gamma0, spread = spread0)
    return m
end

"""
    set_wind!(m; U10, dir)  or  set_wind!(m; u, v)

Uniform wind (dir nautical-from, degrees) or per-cell (nx,ny) components u, v (m/s, blowing towards +x/+y).
"""
function set_wind!(m::WaveModel; U10 = nothing, dir = nothing, u = nothing, v = nothing)
    nx, ny = m.grid.nx, m.grid.ny
    if u !== nothing && v !== nothing
        m.env.u10x .= pad(u, nx, ny); m.env.u10y .= pad(v, nx, ny)
    else
        θ = naut2math(dir)
        m.env.u10x .= U10 * cos(θ); m.env.u10y .= U10 * sin(θ)
    end
    return m
end

"""
    initialize!(m; Hs=0, Tp=10, dir=270, gamma=3.3, spread=10)

Uniform initial sea state in all wet cells (calm if Hs = 0). Resets time to 0.
"""
function initialize!(m::WaveModel; Hs = 0.0, Tp = 10.0, dir = 270.0, gamma = 3.3, spread = 10.0)
    N0 = jonswap(m.spec, Hs, Tp, dir; gamma, spread)
    fill!(m.N, 0.0); fill!(m.N1, 0.0); fill!(m.dN, 0.0)
    nx, ny = m.grid.nx, m.grid.ny
    for j in NG+1:NG+ny, i in NG+1:NG+nx
        m.env.wet[i, j] && (m.N[:, :, i, j] .= N0)
    end
    m.t = 0.0
    apply_bc!(m.N, m, 0.0)
    return m
end

"""
    set_boundaries!(m; left, right, bottom, top)

Each side is `OpenBC()` (default, zero-gradient), `ZeroBC()` (no incoming waves) or `SpectrumBC(...)`.
"""
function set_boundaries!(m::WaveModel; left = m.bc[1], right = m.bc[2], bottom = m.bc[3], top = m.bc[4])
    m.bc = (left, right, bottom, top)
    for (k, b) in enumerate(m.bc)
        if b isa SpectrumBC
            m.Nbc[k] = b.N === nothing ? jonswap(m.spec, b.Hs, b.Tp, b.dir; gamma = b.gamma, spread = b.spread) : b.N
            size(m.Nbc[k]) == (m.spec.ns, m.spec.nth) || error("boundary spectrum must be (ns, nth)")
        end
    end
    apply_bc!(m.N, m, m.t)
    return m
end

"""
    set_environment!(m; depth, U, V, dddt)

Replace depth and/or currents (e.g. from a hydrodynamic model) and refresh derived fields and the time step.
`dddt` (m/s) is the rate of change of depth (tide, surge).
"""
function set_environment!(m::WaveModel; depth = nothing, U = nothing, V = nothing, dddt = nothing)
    nx, ny = m.grid.nx, m.grid.ny
    depth === nothing || (m.env.d .= pad(depth, nx, ny))
    U === nothing || (m.env.U .= pad(U, nx, ny))
    V === nothing || (m.env.V .= pad(V, nx, ny))
    dddt === nothing || (m.env.dddt .= pad(dddt, nx, ny))
    update_environment!(m)
end

# ------------------------------------------------------------------------------------------------
# Environment-derived fields (kinematics are precomputed: the time loop never solves the dispersion relation)
# ------------------------------------------------------------------------------------------------
"""
    update_environment!(m)

Recompute everything that depends only on depth/currents: wet mask, gradients, face group velocities, refraction
and frequency-shift coefficients and the per-cell stable time step. Called automatically by the constructor and
`set_environment!`.
"""
function update_environment!(m::WaveModel)
    m.gsec = nothing                      # face diffusivities of the GSE filter depend on the environment
    e, sp, p, gr = m.env, m.spec, m.params, m.grid
    nx, ny = gr.nx, gr.ny
    ns = sp.ns
    g, hmin = p.g, p.hmin
    for A in (e.d, e.U, e.V, e.dddt)
        fill_ghost!(A)
    end
    @. e.wet = e.d > hmin
    # centred gradients (interior)
    idx2, idy2 = 0.5 / gr.dx, 0.5 / gr.dy
    fill!(e.dddx, 0.0); fill!(e.dddy, 0.0); fill!(e.dUdx, 0.0); fill!(e.dUdy, 0.0); fill!(e.dVdx, 0.0); fill!(e.dVdy, 0.0)
    @threads :static for j in NG+1:NG+ny
        @inbounds for i in NG+1:NG+nx
            e.dddx[i, j] = (e.d[i+1, j] - e.d[i-1, j]) * idx2
            e.dddy[i, j] = (e.d[i, j+1] - e.d[i, j-1]) * idy2
            e.dUdx[i, j] = (e.U[i+1, j] - e.U[i-1, j]) * idx2
            e.dUdy[i, j] = (e.U[i, j+1] - e.U[i, j-1]) * idy2
            e.dVdx[i, j] = (e.V[i+1, j] - e.V[i-1, j]) * idx2
            e.dVdy[i, j] = (e.V[i, j+1] - e.V[i, j-1]) * idy2
        end
    end
    e.has_shift = p.refraction && (any(!iszero, e.U) || any(!iszero, e.V) || any(!iszero, e.dddt))

    fill!(e.cgx, 0.0); fill!(e.cgy, 0.0); fill!(e.ssh, 0.0); fill!(e.csig, 0.0)
    fill!(e.Ufx, 0.0); fill!(e.Vfy, 0.0); fill!(e.wetx, false); fill!(e.wety, false)
    fill!(e.dtcell, Inf)
    fill!(e.kk, 1.0); fill!(e.sh2i, 0.0); fill!(e.gdp, 0.0); fill!(e.gdm, 0.0); fill!(e.gUp, 0.0); fill!(e.gUm, 0.0); fill!(e.refr, false)
    sig, sigf = sp.sig, sp.sigf
    # x faces (left face of cell i, i = NG+1 .. NG+nx+1)
    @threads :static for j in NG+1:NG+ny
        @inbounds for i in NG+1:NG+nx+1
            if e.wet[i-1, j] || e.wet[i, j]
                e.wetx[i, j] = true
                df = 0.5 * (max(e.d[i-1, j], hmin) + max(e.d[i, j], hmin))
                e.Ufx[i, j] = 0.5 * (e.U[i-1, j] + e.U[i, j])
                for is in 1:ns
                    e.cgx[is, i, j] = wavekin(sig[is], df, g)[2]
                end
            end
        end
    end
    # y faces (bottom face of cell j)
    @threads :static for j in NG+1:NG+ny+1
        @inbounds for i in NG+1:NG+nx
            if e.wet[i, j-1] || e.wet[i, j]
                e.wety[i, j] = true
                df = 0.5 * (max(e.d[i, j-1], hmin) + max(e.d[i, j], hmin))
                e.Vfy[i, j] = 0.5 * (e.V[i, j-1] + e.V[i, j])
                for is in 1:ns
                    e.cgy[is, i, j] = wavekin(sig[is], df, g)[2]
                end
            end
        end
    end
    # cell quantities
    @threads :static for j in NG+1:NG+ny
        @inbounds for i in NG+1:NG+nx
            e.wet[i, j] || continue
            d = e.d[i, j]
            gradd = sqrt(e.dddx[i, j]^2 + e.dddy[i, j]^2)
            gradU = sqrt(e.dUdx[i, j]^2 + e.dUdy[i, j]^2 + e.dVdx[i, j]^2 + e.dVdy[i, j]^2)
            rmax = 0.0
            maxssh = 0.0
            for is in 1:ns
                k, cg, dsd = wavekin(sig[is], d, g)
                e.ssh[is, i, j] = dsd / k
                maxssh = max(maxssh, dsd / k)
                e.kk[is, i, j] = k
                e.sh2i[is, i, j] = 1 / sinh(min(k * d, 30.0))^2
                # time step bound (direction independent), see wavecelldt in the CUDA version
                r = cg * sqrt(1 / gr.dx^2 + 1 / gr.dy^2) + abs(e.U[i, j]) / gr.dx + abs(e.V[i, j]) / gr.dy
                if p.refraction
                    r += (dsd / k * gradd + gradU) / sp.dth
                    cs = dsd * (abs(e.dddt[i, j]) + abs(e.U[i, j] * e.dddx[i, j]) + abs(e.V[i, j] * e.dddy[i, j])) + cg * k * gradU
                    r += cs / sp.dsig[is]
                end
                rmax = max(rmax, r)
            end
            e.dtcell[i, j] = rmax > 0 ? 1 / rmax : Inf
            # refraction coefficients, computed once: c_θ(face) = ssh·gd + gU  with
            #   gd = s ∂d/∂x − c ∂d/∂y ,  gU = c s ∂U/∂x − c² ∂U/∂y + s² ∂V/∂x − c s ∂V/∂y   at the face direction (c, s)
            if p.refraction && maxssh * gradd + gradU + abs(e.dddt[i, j]) > 1e-14
                e.refr[i, j] = true
                dddx, dddy, Ux, Uy, Vx, Vy = e.dddx[i, j], e.dddy[i, j], e.dUdx[i, j], e.dUdy[i, j], e.dVdx[i, j], e.dVdy[i, j]
                for it in 1:sp.nth
                    c0, s0 = sp.cth[it], sp.sth[it]
                    cp, spf = c0 * sp.ch - s0 * sp.sh, s0 * sp.ch + c0 * sp.sh
                    cm, smf = c0 * sp.ch + s0 * sp.sh, s0 * sp.ch - c0 * sp.sh
                    e.gdp[it, i, j] = spf * dddx - cp * dddy
                    e.gUp[it, i, j] = cp * spf * Ux - cp * cp * Uy + spf * spf * Vx - cp * spf * Vy
                    e.gdm[it, i, j] = smf * dddx - cm * dddy
                    e.gUm[it, i, j] = cm * smf * Ux - cm * cm * Uy + smf * smf * Vx - cm * smf * Vy
                end
            end
            if e.has_shift
                for f in 1:ns+1
                    kf, cgf, dsf = wavekin(sigf[f], d, g)
                    e.csig[1, f, i, j] = dsf * (e.dddt[i, j] + e.U[i, j] * e.dddx[i, j] + e.V[i, j] * e.dddy[i, j])
                    e.csig[2, f, i, j] = -cgf * kf * e.dUdx[i, j]
                    e.csig[3, f, i, j] = -cgf * kf * (e.dUdy[i, j] + e.dVdx[i, j])
                    e.csig[4, f, i, j] = -cgf * kf * e.dVdy[i, j]
                end
            end
        end
    end
    return m
end

"""
    stable_dt(m, cfl=0.4)

Largest stable time step (CFL on the summed x, y, θ, σ advection rates, maximised over frequency).
The CFL is capped at 0.5 (midpoint predictor/corrector with a limiter).
"""
stable_dt(m::WaveModel, cfl = 0.4) = min(cfl, 0.5) * minimum(@view m.env.dtcell[NG+1:NG+m.grid.nx, NG+1:NG+m.grid.ny])

# ------------------------------------------------------------------------------------------------
# Boundary conditions (ghost cells)
# ------------------------------------------------------------------------------------------------
function apply_bc!(N::Array{Float64,4}, m::WaveModel, t)
    nx, ny = m.grid.nx, m.grid.ny
    nxg, nyg = nx + 2NG, ny + 2NG
    bc = m.bc
    @inline function val(k)
        b = bc[k]
        s = b.scale(t)
        return s * s .* m.Nbc[k]
    end
    # left / right over interior rows
    for (k, side) in ((1, :left), (2, :right))
        if bc[k] isa SpectrumBC
            v = val(k)
            for j in 1:nyg, g in 1:NG
                ig = side === :left ? g : nxg - g + 1
                N[:, :, ig, j] .= v
            end
        elseif bc[k] isa ZeroBC
            for j in 1:nyg, g in 1:NG
                N[:, :, side === :left ? g : nxg - g + 1, j] .= 0.0
            end
        else
            for j in 1:nyg, g in 1:NG
                ig, ii = side === :left ? (g, NG + 1) : (nxg - g + 1, nxg - NG)
                N[:, :, ig, j] .= @view N[:, :, ii, j]
            end
        end
    end
    for (k, side) in ((3, :bottom), (4, :top))
        if bc[k] isa SpectrumBC
            v = val(k)
            for i in 1:nxg, g in 1:NG
                jg = side === :bottom ? g : nyg - g + 1
                N[:, :, i, jg] .= v
            end
        elseif bc[k] isa ZeroBC
            for i in 1:nxg, g in 1:NG
                N[:, :, i, side === :bottom ? g : nyg - g + 1] .= 0.0
            end
        else
            for i in 1:nxg, g in 1:NG
                jg, jj = side === :bottom ? (g, NG + 1) : (nyg - g + 1, nyg - NG)
                N[:, :, i, jg] .= @view N[:, :, i, jj]
            end
        end
    end
    return N
end

# ------------------------------------------------------------------------------------------------
# Tendency  dN/dt = L(N)
# ------------------------------------------------------------------------------------------------
@inline function fluxx(N, is, it, i, j, cx, θ, order)
    @inbounds begin
        Nl = N[is, it, i-1, j]; Nr = N[is, it, i, j]
        if cx > 0
            sl = order > 1 ? limslope(N[is, it, i-2, j], Nl, Nr, θ) : 0.0
            return cx * (Nl + 0.5 * sl)
        else
            sr = order > 1 ? limslope(Nl, Nr, N[is, it, i+1, j], θ) : 0.0
            return cx * (Nr - 0.5 * sr)
        end
    end
end
@inline function fluxy(N, is, it, i, j, cy, θ, order)
    @inbounds begin
        Nb = N[is, it, i, j-1]; Nt = N[is, it, i, j]
        if cy > 0
            sb = order > 1 ? limslope(N[is, it, i, j-2], Nb, Nt, θ) : 0.0
            return cy * (Nb + 0.5 * sb)
        else
            st = order > 1 ? limslope(Nb, Nt, N[is, it, i, j+1], θ) : 0.0
            return cy * (Nt - 0.5 * st)
        end
    end
end

"""
Tendency of every interior cell: physical-space flux divergence + spectral (θ, σ) transport.

Loops over directions; for each direction (1) the x and y face fluxes of all frequencies are computed once into
scratch arrays (instead of twice, once from each side of a face), (2) the divergence and the spectral terms are added.
Both passes are threaded over rows and write disjoint data, so the result is identical for any number of threads and
conservative to round-off. The refraction coefficients are read from the arrays prepared by `update_environment!`
(and skipped in cells where they are exactly zero, e.g. deep water without currents).
"""
function tendency!(dN, N, m::WaveModel)
    sp, p, e, gr = m.spec, m.params, m.env, m.grid
    ns, nth = sp.ns, sp.nth
    nx, ny = gr.nx, gr.ny
    order, θl = p.order, p.theta
    idx, idy = 1 / gr.dx, 1 / gr.dy
    sig, sigf, dsig, cth, sth, rdp, rdc = sp.sig, sp.sigf, sp.dsig, sp.cth, sp.sth, sp.rdp, sp.rdc
    dth, idth = sp.dth, 1 / sp.dth
    doshift = e.has_shift
    Fx, Fy = e.Fx, e.Fy
    o2 = order > 1 ? 1.0 : 0.0
    @inline slope_nu(Nm, N0, Np, rl, rr, rc) = minmod3(θl * (N0 - Nm) * rl, (Np - Nm) * rc, θl * (Np - N0) * rr)
    for it in 1:nth
        c0, s0 = cth[it], sth[it]
        itp = it == nth ? 1 : it + 1
        itm = it == 1 ? nth : it - 1
        itpp = itp == nth ? 1 : itp + 1
        itmm = itm == 1 ? nth : itm - 1
        # ---- pass A: face fluxes of this direction (x faces: left face of cell i; y faces: bottom face of cell j)
        @threads :static for j in NG+1:NG+ny+1
            @inbounds begin
                if j <= NG + ny
                    for i in NG+1:NG+nx+1
                        if e.wetx[i, j]
                            Ufl = e.Ufx[i, j]
                            @simd for is in 1:ns
                                cx = e.cgx[is, i, j] * c0 + Ufl
                                Nl = N[is, it, i-1, j]; Nr = N[is, it, i, j]
                                sl = o2 * limslope(N[is, it, i-2, j], Nl, Nr, θl)
                                sr = o2 * limslope(Nl, Nr, N[is, it, i+1, j], θl)
                                Fx[is, i, j] = ifelse(cx > 0, cx * (Nl + 0.5 * sl), cx * (Nr - 0.5 * sr))
                            end
                        else
                            for is in 1:ns
                                Fx[is, i, j] = 0.0
                            end
                        end
                    end
                end
                for i in NG+1:NG+nx
                    if e.wety[i, j]
                        Vfb = e.Vfy[i, j]
                        @simd for is in 1:ns
                            cy = e.cgy[is, i, j] * s0 + Vfb
                            Nb = N[is, it, i, j-1]; Nt = N[is, it, i, j]
                            sb = o2 * limslope(N[is, it, i, j-2], Nb, Nt, θl)
                            st = o2 * limslope(Nb, Nt, N[is, it, i, j+1], θl)
                            Fy[is, i, j] = ifelse(cy > 0, cy * (Nb + 0.5 * sb), cy * (Nt - 0.5 * st))
                        end
                    else
                        for is in 1:ns
                            Fy[is, i, j] = 0.0
                        end
                    end
                end
            end
        end
        # ---- pass B: divergence + spectral transport
        @threads :static for j in NG+1:NG+ny
            @inbounds for i in NG+1:NG+nx
                if !e.wet[i, j]
                    for is in 1:ns
                        dN[is, it, i, j] = 0.0
                    end
                    continue
                end
                dorefr = e.refr[i, j]
                gdp, gdm, gUp, gUm = e.gdp[it, i, j], e.gdm[it, i, j], e.gUp[it, i, j], e.gUm[it, i, j]
                if dorefr
                    @simd for is in 1:ns
                        div = (Fx[is, i+1, j] - Fx[is, i, j]) * idx + (Fy[is, i, j+1] - Fy[is, i, j]) * idy
                        Nc = N[is, it, i, j]
                        ssh = e.ssh[is, i, j]
                        Np = N[is, itp, i, j]; Nm = N[is, itm, i, j]
                        sc = o2 * limslope(Nm, Nc, Np, θl)
                        spl = o2 * limslope(Nc, Np, N[is, itpp, i, j], θl)
                        sm = o2 * limslope(N[is, itmm, i, j], Nm, Nc, θl)
                        cthp = ssh * gdp + gUp
                        cthm = ssh * gdm + gUm
                        Fp = ifelse(cthp > 0, cthp * (Nc + 0.5 * sc), cthp * (Np - 0.5 * spl))
                        Fm = ifelse(cthm > 0, cthm * (Nm + 0.5 * sm), cthm * (Nc - 0.5 * sc))
                        dN[is, it, i, j] = -(div + (Fp - Fm) * idth)
                    end
                else
                    @simd for is in 1:ns
                        dN[is, it, i, j] = -((Fx[is, i+1, j] - Fx[is, i, j]) * idx + (Fy[is, i, j+1] - Fy[is, i, j]) * idy)
                    end
                end
                if doshift && dorefr
                    for is in 1:ns
                        div = 0.0
                        σ = sig[is]
                        Nc = N[is, it, i, j]
                        csp = e.csig[1, is+1, i, j] + e.csig[2, is+1, i, j] * c0 * c0 + e.csig[3, is+1, i, j] * c0 * s0 + e.csig[4, is+1, i, j] * s0 * s0
                        csm = e.csig[1, is, i, j] + e.csig[2, is, i, j] * c0 * c0 + e.csig[3, is, i, j] * c0 * s0 + e.csig[4, is, i, j] * s0 * s0
                        slc = 0.0
                        if order > 1 && is > 1 && is < ns
                            slc = slope_nu(N[is-1, it, i, j], Nc, N[is+1, it, i, j], rdp[is-1], rdp[is], rdc[is])
                        end
                        Fp = 0.0
                        if is == ns
                            Fp = csp > 0 ? csp * Nc : 0.0
                        else
                            Nn = N[is+1, it, i, j]
                            sln = 0.0
                            if order > 1 && is + 1 < ns
                                sln = slope_nu(Nc, Nn, N[is+2, it, i, j], rdp[is], rdp[is+1], rdc[is+1])
                            end
                            Fp = csp > 0 ? csp * (Nc + slc * (sigf[is+1] - σ)) : csp * (Nn + sln * (sigf[is+1] - sig[is+1]))
                        end
                        Fm = 0.0
                        if is > 1
                            Nl = N[is-1, it, i, j]
                            sll = 0.0
                            if order > 1 && is - 1 > 1
                                sll = slope_nu(N[is-2, it, i, j], Nl, Nc, rdp[is-2], rdp[is-1], rdc[is-1])
                            end
                            Fm = csm > 0 ? csm * (Nl + sll * (sigf[is] - sig[is-1])) : csm * (Nc + slc * (sigf[is] - σ))
                        end
                        div += (Fp - Fm) / dsig[is]
                        dN[is, it, i, j] -= div
                    end
                end
            end
        end
    end
    return dN
end

# ------------------------------------------------------------------------------------------------
# Source terms (semi-implicit)
# ------------------------------------------------------------------------------------------------
"""
    breakingfraction(b)

Fraction of breaking waves Qb solving (1-Qb)/(-ln Qb) = b², b = Hrms/Hmax (Battjes & Janssen 1978).
"""
function breakingfraction(b)
    b >= 1 && return 1.0
    b < 0.2 && return 0.0
    lo, hi, b2 = 1e-12, 1 - 1e-7, b * b
    for _ in 1:40
        mid = 0.5 * (lo + hi)
        if 1 - mid + b2 * log(mid) < 0
            lo = mid
        else
            hi = mid
        end
    end
    return 0.5 * (lo + hi)
end

"""
    cell_source!(Nc, p, spec, d, u10x, u10y, dt, kk)

Apply wind input, whitecapping, depth-induced breaking and bottom friction to one cell's (ns, nth) spectrum:
N ← (N + min(dt (A/σ + B N), ΔN_max)) / (1 + dt (D_wc + D_br + D_bf)). Non-negative for any dt.
"""
function cell_source!(Nc::AbstractMatrix, p::WaveParams, sp::SpectralGrid, d, u10x, u10y, dt, kk::AbstractVector, sh2i::AbstractVector)
    ns, nth, g = sp.ns, sp.nth, p.g
    sig, dsig, cth, sth, dth = sp.sig, sp.dsig, sp.cth, sp.sth, sp.dth
    Etot = 0.0; m1 = 0.0; mm1 = 0.0; sk = 0.0
    @inbounds for is in 1:ns
        σ = sig[is]
        s = 0.0
        for it in 1:nth
            s += Nc[is, it]
        end
        Es = σ * s * dth * dsig[is]
        Etot += Es; m1 += σ * Es; mm1 += Es / σ; sk += sqrt(kk[is]) * Es
    end
    haveE = Etot > 1e-12
    sbar = haveE ? m1 / Etot : 1.0
    stil = haveE ? Etot / mm1 : 1.0
    ktil = haveE ? (sk / Etot)^2 : 1.0
    steep = ktil * sqrt(Etot)

    brate = 0.0
    if p.breaking && haveE
        Hm = p.gamma_bj * d
        Qb = breakingfraction(sqrt(8Etot) / Hm)
        brate = 0.25 * p.alpha_bj * Qb * (sbar / 2π) * Hm^2 / Etot
    end
    wcfac = (p.whitecap && haveE) ? p.Cds * (steep / sqrt(p.stpm))^p.p_ds * stil : 0.0

    U10 = hypot(u10x, u10y)
    dowind = p.wind && U10 > 0.1
    ustar = 0.0; cw = 1.0; sw = 0.0; σPM = 1.0
    if dowind
        Cd = U10 < 7.5 ? 1.2875e-3 : (0.8 + 0.065U10) * 1e-3
        ustar = U10 * sqrt(Cd)
        cw, sw = u10x / U10, u10y / U10
        σPM = 2π * 0.13 * g / (28ustar)
    end
    @inbounds for is in 1:ns
        σ = sig[is]; k = kk[is]
        dec = brate
        p.whitecap && haveE && (dec += wcfac * ((1 - p.delta_ds) + p.delta_ds * k / ktil) * (k / ktil))
        p.friction && (dec += p.Cbf * σ^2 / g^2 * sh2i[is])
        cph = σ / k
        Alin = 0.0; dNmax = 0.0
        if dowind
            σ2 = σ * σ
            Alin = 1.5e-3 / (2π * g^2) * exp(-(σ / σPM)^-4)
            dNmax = p.Cl * p.alpha_pm * g^2 / (2π * σ2 * σ2 * σ2)
        end
        for it in 1:nth
            N = Nc[is, it]
            dNw = 0.0
            if dowind
                cosd = cth[it] * cw + sth[it] * sw
                uc = ustar * max(0.0, cosd)
                A = Alin * uc^4
                B = max(0.0, 0.25 * p.rhoratio * (28ustar / cph * cosd - 1)) * σ
                dNw = min(dt * (A / σ + B * N), dNmax)
            end
            Nc[is, it] = (N + dNw) / (1 + dt * dec)
        end
    end
    return Nc
end

function sources!(m::WaveModel, dt)
    p = m.params
    (p.wind || p.whitecap || p.breaking || p.friction) || return m
    nx, ny = m.grid.nx, m.grid.ny
    e, sp = m.env, m.spec
    @threads :static for j in NG+1:NG+ny
        for i in NG+1:NG+nx
            e.wet[i, j] || continue
            cell_source!(view(m.N, :, :, i, j), p, sp, e.d[i, j], e.u10x[i, j], e.u10y[i, j], dt,
                         view(e.kk, :, i, j), view(e.sh2i, :, i, j))
        end
    end
    return m
end

# ------------------------------------------------------------------------------------------------
# Garden sprinkler effect alleviation (anisotropic diffusion of each spectral bin)
# ------------------------------------------------------------------------------------------------
"""
    set_gse!(m; alpha_s = 3.0, alpha_n = 3.0, cfl = 1.0, tau = 0.0)
    set_gse!(m, nothing)

Switch garden-sprinkler-effect alleviation on (see [`GSEParams`](@ref)) or off. Works with `step!`/`run!` and with the
implicit solver of `WaveSweep.jl`.
"""
function set_gse!(m::WaveModel; alpha_s = 3.0, alpha_n = 3.0, cfl = 1.0, tau = 0.0, maxsub = 400, safety = 2.5, adi_above = 2)
    m.gse = GSEParams(; alpha_s = Float64(alpha_s), alpha_n = Float64(alpha_n), cfl = Float64(cfl), tau = Float64(tau), maxsub = Int(maxsub),
                      safety = Float64(safety), adi_above = Int(adi_above))
    m.gsec = nothing
    return m
end
set_gse!(m::WaveModel, ::Nothing) = (m.gse = nothing; m.gsec = nothing; m)

# Face diffusivities of the filter, built once per environment (they depend only on depth/currents and the GSE settings):
#   Dss = α_s²/3 (Δcg)² Ts ,  Dnn = α_n²/3 (cg Δθ)² Ts   at every x face (left face of cell i) and y face (bottom face of cell j),
# for every frequency.  In `Ts = dt` mode the arrays are stored per unit Ts and scaled by dt at use.
mutable struct GSECache
    Dsx::Array{Float64,3}; Dnx::Array{Float64,3}; Dsy::Array{Float64,3}; Dny::Array{Float64,3}
    gwx::Matrix{Bool}; gwy::Matrix{Bool}      # face between two wet cells (diffusive flux allowed)
    dtmode::Bool
    Dsum::Float64                             # max over faces of Dss + Dnn
    Gc::Array{Float64,3}                      # scratch (ADI)
end

function gse_prepare!(m::WaveModel)
    g, sp, e, gr = m.gse, m.spec, m.env, m.grid
    ns = sp.ns; nx, ny = gr.nx, gr.ny
    nxg, nyg = nx + 2NG, ny + 2NG
    dxm = min(gr.dx, gr.dy); kdc = 0.5 * (sp.fac - 1 / sp.fac); dth = sp.dth
    dtmode = g.tau <= 0 && g.cfl <= 0
    Dsx = zeros(ns, nxg, nyg); Dnx = zeros(ns, nxg, nyg); Dsy = zeros(ns, nxg, nyg); Dny = zeros(ns, nxg, nyg)
    gwx = fill(false, nxg, nyg); gwy = fill(false, nxg, nyg)
    Dsum = 0.0
    as, an = g.alpha_s^2 / 3, g.alpha_n^2 / 3
    Ts(cg) = dtmode ? 1.0 : (g.tau > 0 ? g.tau : g.cfl * dxm / cg)
    for j in NG+1:NG+ny, i in NG+1:NG+nx+1
        (e.wet[i-1, j] && e.wet[i, j]) || continue
        gwx[i, j] = true
        for is in 1:ns
            cg = e.cgx[is, i, j]; τ = Ts(cg)
            Dsx[is, i, j] = as * (kdc * cg)^2 * τ; Dnx[is, i, j] = an * (cg * dth)^2 * τ
            Dsum = max(Dsum, Dsx[is, i, j] + Dnx[is, i, j])
        end
    end
    for j in NG+1:NG+ny+1, i in NG+1:NG+nx
        (e.wet[i, j-1] && e.wet[i, j]) || continue
        gwy[i, j] = true
        for is in 1:ns
            cg = e.cgy[is, i, j]; τ = Ts(cg)
            Dsy[is, i, j] = as * (kdc * cg)^2 * τ; Dny[is, i, j] = an * (cg * dth)^2 * τ
            Dsum = max(Dsum, Dsy[is, i, j] + Dny[is, i, j])
        end
    end
    m.gsec = GSECache(Dsx, Dnx, Dsy, Dny, gwx, gwy, dtmode, Dsum, zeros(ns, nxg, nyg))
    return m.gsec
end

# One explicit forward-Euler sub-step of ∂N/∂τ = ∇·(D∇N), all directions. Directions are independent (the filter only couples
# neighbouring cells of the same bin), so per direction: (A) face fluxes (each computed once, vectorised over frequency),
# (B) in-place update of the cell values. Both passes are threaded over rows and write disjoint data.
function gse_explicit_substep!(m::WaveModel, c::GSECache, dτ, sc, clip::Bool)
    sp, e, gr, N = m.spec, m.env, m.grid, m.N
    ns, nth, nx, ny = sp.ns, sp.nth, gr.nx, gr.ny
    idx, idy = 1 / gr.dx, 1 / gr.dy
    Fx, Fy = e.Fx, e.Fy
    for it in 1:nth
        c0, s0 = sp.cth[it], sp.sth[it]
        cc, ss, cs = c0 * c0, s0 * s0, c0 * s0
        @threads :static for j in NG+1:NG+ny+1
            @inbounds begin
                if j <= NG + ny
                    for i in NG+1:NG+nx+1
                        if c.gwx[i, j]
                            @simd for is in 1:ns
                                Dss = c.Dsx[is, i, j] * sc; Dnn = c.Dnx[is, i, j] * sc
                                gy = 0.25 * idy * ((N[is, it, i-1, j+1] - N[is, it, i-1, j-1]) + (N[is, it, i, j+1] - N[is, it, i, j-1]))
                                Fx[is, i, j] = -((Dss * cc + Dnn * ss) * (N[is, it, i, j] - N[is, it, i-1, j]) * idx + (Dss - Dnn) * cs * gy)
                            end
                        else
                            for is in 1:ns; Fx[is, i, j] = 0.0; end
                        end
                    end
                end
                for i in NG+1:NG+nx
                    if c.gwy[i, j]
                        @simd for is in 1:ns
                            Dss = c.Dsy[is, i, j] * sc; Dnn = c.Dny[is, i, j] * sc
                            gx = 0.25 * idx * ((N[is, it, i+1, j-1] - N[is, it, i-1, j-1]) + (N[is, it, i+1, j] - N[is, it, i-1, j]))
                            Fy[is, i, j] = -((Dss - Dnn) * cs * gx + (Dss * ss + Dnn * cc) * (N[is, it, i, j] - N[is, it, i, j-1]) * idy)
                        end
                    else
                        for is in 1:ns; Fy[is, i, j] = 0.0; end
                    end
                end
            end
        end
        @threads :static for j in NG+1:NG+ny
            @inbounds for i in NG+1:NG+nx
                e.wet[i, j] || continue
                if clip
                    @simd for is in 1:ns
                        N[is, it, i, j] = max(N[is, it, i, j] - dτ * ((Fx[is, i+1, j] - Fx[is, i, j]) * idx + (Fy[is, i, j+1] - Fy[is, i, j]) * idy), 0.0)
                    end
                else
                    @simd for is in 1:ns
                        N[is, it, i, j] -= dτ * ((Fx[is, i+1, j] - Fx[is, i, j]) * idx + (Fy[is, i, j+1] - Fy[is, i, j]) * idy)
                    end
                end
            end
        end
    end
    return m
end

# One unconditionally stable step of length dt: Douglas-Gunn ADI. The along-axis diffusion (Dxx in x, Dyy in y) is implicit
# (tridiagonal solves along rows and columns, vectorised over frequency), the cross term Dxy is explicit.
#   (I - dt Lx) N* = (I + dt (Ly + Lc)) N ;   (I - dt Ly) N' = N* - dt Ly N        (conservative; first order in time)
# Faces towards dry cells carry no flux; zero-gradient (OpenBC) boundaries carry no flux; imposed / zero-inflow ghost cells are
# known (lagged) Dirichlet values. Cost is independent of dt, unlike the explicit sub-stepping.
function gse_adi_step!(m::WaveModel, c::GSECache, dt, sc)
    sp, e, gr, N = m.spec, m.env, m.grid, m.N
    ns, nth, nx, ny = sp.ns, sp.nth, gr.nx, gr.ny
    idx, idy = 1 / gr.dx, 1 / gr.dy
    idx2, idy2 = idx * idx, idy * idy
    Fx, Fy, Gc = e.Fx, e.Fy, c.Gc
    nmax = max(nx, ny)
    cps = [zeros(ns, nmax) for _ in 1:Threads.maxthreadid()]
    openL, openR, openB, openT = m.bc[1] isa OpenBC, m.bc[2] isa OpenBC, m.bc[3] isa OpenBC, m.bc[4] isa OpenBC
    for it in 1:nth
        c0, s0 = sp.cth[it], sp.sth[it]
        cc, ss, cs = c0 * c0, s0 * s0, c0 * s0
        # (A) explicit pieces from N: cross flux in x (Fx), axis flux in y (Fy), cross flux in y (Gc)
        @threads :static for j in NG+1:NG+ny+1
            @inbounds begin
                if j <= NG + ny
                    for i in NG+1:NG+nx+1
                        if c.gwx[i, j]
                            @simd for is in 1:ns
                                Dss = c.Dsx[is, i, j] * sc; Dnn = c.Dnx[is, i, j] * sc
                                gy = 0.25 * idy * ((N[is, it, i-1, j+1] - N[is, it, i-1, j-1]) + (N[is, it, i, j+1] - N[is, it, i, j-1]))
                                Fx[is, i, j] = -(Dss - Dnn) * cs * gy
                            end
                        else
                            for is in 1:ns; Fx[is, i, j] = 0.0; end
                        end
                    end
                end
                for i in NG+1:NG+nx
                    if c.gwy[i, j]
                        @simd for is in 1:ns
                            Dss = c.Dsy[is, i, j] * sc; Dnn = c.Dny[is, i, j] * sc
                            gx = 0.25 * idx * ((N[is, it, i+1, j-1] - N[is, it, i-1, j-1]) + (N[is, it, i+1, j] - N[is, it, i-1, j]))
                            Gc[is, i, j] = -(Dss - Dnn) * cs * gx
                            Fy[is, i, j] = -(Dss * ss + Dnn * cc) * (N[is, it, i, j] - N[is, it, i, j-1]) * idy
                        end
                    else
                        for is in 1:ns; Gc[is, i, j] = 0.0; Fy[is, i, j] = 0.0; end
                    end
                end
            end
        end
        # (B) right-hand side of the x-implicit stage: N + dt (Lc + Ly) N
        @threads :static for j in NG+1:NG+ny
            @inbounds for i in NG+1:NG+nx
                e.wet[i, j] || continue
                @simd for is in 1:ns
                    N[is, it, i, j] -= dt * ((Fx[is, i+1, j] - Fx[is, i, j]) * idx + (Fy[is, i, j+1] - Fy[is, i, j] + Gc[is, i, j+1] - Gc[is, i, j]) * idy)
                end
            end
        end
        # (C) implicit solves along x (one row per task)
        @threads :static for j in NG+1:NG+ny
            cp = cps[Threads.threadid()]
            @inbounds begin
                for q in 1:nx
                    i = NG + q
                    Dlv = c.gwx[i, j] && !(q == 1 && openL); Drv = c.gwx[i+1, j] && !(q == nx && openR)
                    @simd for is in 1:ns
                        Dl = Dlv ? (c.Dsx[is, i, j] * cc + c.Dnx[is, i, j] * ss) * sc * dt * idx2 : 0.0
                        Dr = Drv ? (c.Dsx[is, i+1, j] * cc + c.Dnx[is, i+1, j] * ss) * sc * dt * idx2 : 0.0
                        rhs = N[is, it, i, j] + (q == 1 ? Dl * N[is, it, i-1, j] : 0.0) + (q == nx ? Dr * N[is, it, i+1, j] : 0.0)
                        a = q == 1 ? 0.0 : -Dl
                        bet = (1.0 + Dl + Dr) - (q == 1 ? 0.0 : a * cp[is, q-1])
                        cp[is, q] = (q == nx ? 0.0 : -Dr) / bet
                        N[is, it, i, j] = (rhs - (q == 1 ? 0.0 : a * N[is, it, i-1, j])) / bet
                    end
                end
                for q in nx-1:-1:1
                    i = NG + q
                    @simd for is in 1:ns
                        N[is, it, i, j] -= cp[is, q] * N[is, it, i+1, j]
                    end
                end
            end
        end
        # (D) right-hand side of the y-implicit stage: N* - dt Ly N
        @threads :static for j in NG+1:NG+ny
            @inbounds for i in NG+1:NG+nx
                e.wet[i, j] || continue
                @simd for is in 1:ns
                    N[is, it, i, j] += dt * (Fy[is, i, j+1] - Fy[is, i, j]) * idy
                end
            end
        end
        # (E) implicit solves along y (one column per task), clipping negatives in the back substitution
        @threads :static for i in NG+1:NG+nx
            cp = cps[Threads.threadid()]
            @inbounds begin
                for q in 1:ny
                    j = NG + q
                    Dlv = c.gwy[i, j] && !(q == 1 && openB); Drv = c.gwy[i, j+1] && !(q == ny && openT)
                    @simd for is in 1:ns
                        Dl = Dlv ? (c.Dsy[is, i, j] * ss + c.Dny[is, i, j] * cc) * sc * dt * idy2 : 0.0
                        Dr = Drv ? (c.Dsy[is, i, j+1] * ss + c.Dny[is, i, j+1] * cc) * sc * dt * idy2 : 0.0
                        rhs = N[is, it, i, j] + (q == 1 ? Dl * N[is, it, i, j-1] : 0.0) + (q == ny ? Dr * N[is, it, i, j+1] : 0.0)
                        a = q == 1 ? 0.0 : -Dl
                        bet = (1.0 + Dl + Dr) - (q == 1 ? 0.0 : a * cp[is, q-1])
                        cp[is, q] = (q == ny ? 0.0 : -Dr) / bet
                        N[is, it, i, j] = (rhs - (q == 1 ? 0.0 : a * N[is, it, i, j-1])) / bet
                    end
                end
                for q in ny-1:-1:1
                    j = NG + q
                    @simd for is in 1:ns
                        N[is, it, i, j] -= cp[is, q] * N[is, it, i, j+1]
                    end
                end
                for q in 1:ny
                    j = NG + q
                    @simd for is in 1:ns
                        N[is, it, i, j] = max(N[is, it, i, j], 0.0)
                    end
                end
            end
        end
    end
    return m
end

"""
    gse_step!(m, dt)

Apply the garden-sprinkler filter over one solver step of length `dt`. Returns the number of explicit sub-steps used, or 0 if
the filter is off, or -1 if the unconditionally stable ADI step was used (chosen when the explicit scheme would need more than
`adi_above` sub-steps, i.e. for large implicit time steps).
"""
function gse_step!(m::WaveModel, dt)
    g = m.gse
    g === nothing && return 0
    c = m.gsec === nothing ? gse_prepare!(m) : m.gsec
    gr = m.grid
    sc = c.dtmode ? dt : 1.0
    dxm = min(gr.dx, gr.dy)
    nsub = max(1, ceil(Int, g.safety * dt * c.Dsum * sc / dxm^2))
    apply_bc!(m.N, m, m.t)
    if nsub > g.adi_above
        gse_adi_step!(m, c, dt, sc)
        return -1
    end
    nsub = min(nsub, g.maxsub)
    dτ = dt / nsub
    for k in 1:nsub
        k > 1 && apply_bc!(m.N, m, m.t)
        gse_explicit_substep!(m, c, dτ, sc, k == nsub)
    end
    return nsub
end

# ------------------------------------------------------------------------------------------------
# Time stepping
# ------------------------------------------------------------------------------------------------
# out = max(N0 + dt dN, 0) on wet interior cells, 0 on dry ones
function advance!(out, N0, dN, dt, m::WaveModel)
    e = m.env
    nx, ny = m.grid.nx, m.grid.ny
    @threads :static for j in NG+1:NG+ny
        @inbounds for i in NG+1:NG+nx
            if e.wet[i, j]
                for it in 1:m.spec.nth, is in 1:m.spec.ns
                    out[is, it, i, j] = max(N0[is, it, i, j] + dt * dN[is, it, i, j], 0.0)
                end
            else
                for it in 1:m.spec.nth, is in 1:m.spec.ns
                    out[is, it, i, j] = 0.0
                end
            end
        end
    end
    return out
end

"""
    step!(m, dt)

One predictor/corrector step: half step into N1, full step from N using the tendency at N1, then sources.
"""
function step!(m::WaveModel, dt)
    apply_bc!(m.N, m, m.t)
    tendency!(m.dN, m.N, m)
    advance!(m.N1, m.N, m.dN, 0.5dt, m)
    apply_bc!(m.N1, m, m.t + 0.5dt)
    tendency!(m.dN, m.N1, m)
    advance!(m.N, m.N, m.dN, dt, m)
    sources!(m, dt)
    m.t += dt
    gse_step!(m, dt)
    return m
end

"""
    run!(m; tend, cfl=0.4, dt=nothing, output_dt=nothing, callback=nothing, verbose=true)

Advance to absolute time `tend` (seconds). The time step is `stable_dt(m, cfl)` unless `dt` is given, and is
shortened to hit `tend` and every `output_dt`. `callback(m, t)` is called at the start and at each output time.
Returns the number of steps taken.
"""
function run!(m::WaveModel; tend, cfl = 0.4, dt = nothing, output_dt = nothing, callback = nothing, verbose = true)
    dtmax = dt === nothing ? stable_dt(m, cfl) : dt
    isfinite(dtmax) || error("no wet cells: nothing to compute")
    verbose && println("WaveAction: dt = $(round(dtmax, digits = 4)) s, threads = $(nthreads()), grid $(m.grid.nx)×$(m.grid.ny), spectrum $(m.spec.ns)×$(m.spec.nth)")
    nextout = output_dt === nothing ? Inf : m.t + output_dt
    callback === nothing || callback(m, m.t)
    nsteps = 0
    tol = 1e-9 * max(1.0, abs(tend))
    while m.t < tend - tol
        h = min(dtmax, tend - m.t, nextout - m.t)
        step!(m, h)
        nsteps += 1
        if m.t >= nextout - tol
            callback === nothing || callback(m, m.t)
            nextout += output_dt
        end
        if !all(isfinite, @view m.N[:, :, NG+1:NG+m.grid.nx, NG+1:NG+m.grid.ny])
            error("non-finite wave action at t = $(m.t) s")
        end
    end
    return nsteps
end

# ------------------------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------------------------
"""
    bulk(m) -> (Hs, Tm, Dir, Sxx, Sxy, Syy)

(nx, ny) fields. Hs = 4√m0, Tm = 2π m0/m1, Dir mean direction (nautical, coming from, degrees),
radiation stresses in N/m (scaled by `params.rhog`).
"""
function bulk(m::WaveModel)
    sp, p, e = m.spec, m.params, m.env
    nx, ny = m.grid.nx, m.grid.ny
    Hs = zeros(nx, ny); Tm = zeros(nx, ny); Dir = zeros(nx, ny)
    Sxx = zeros(nx, ny); Sxy = zeros(nx, ny); Syy = zeros(nx, ny)
    @threads :static for j in 1:ny
        nloc = zeros(sp.ns)
        for i in 1:nx
            ig, jg = i + NG, j + NG
            e.wet[ig, jg] || continue
            for is in 1:sp.ns
                k, cg, _ = wavekin(sp.sig[is], e.d[ig, jg], p.g)
                nloc[is] = cg * k / sp.sig[is]
            end
            m0 = 0.0; m1 = 0.0; cc = 0.0; ss = 0.0; sxx = 0.0; sxy = 0.0; syy = 0.0
            for it in 1:sp.nth, is in 1:sp.ns
                E = sp.sig[is] * m.N[is, it, ig, jg] * sp.dth * sp.dsig[is]
                c, s = sp.cth[it], sp.sth[it]
                m0 += E; m1 += sp.sig[is] * E; cc += c * E; ss += s * E
                sxx += (nloc[is] * (c * c + 1) - 0.5) * E
                sxy += nloc[is] * s * c * E
                syy += (nloc[is] * (s * s + 1) - 0.5) * E
            end
            Hs[i, j] = 4sqrt(m0)
            Tm[i, j] = m1 > 1e-20 ? 2π * m0 / m1 : 0.0
            Dir[i, j] = m0 > 1e-20 ? math2naut(atan(ss, cc)) : 0.0
            Sxx[i, j] = p.rhog * sxx; Sxy[i, j] = p.rhog * sxy; Syy[i, j] = p.rhog * syy
        end
    end
    return (Hs = Hs, Tm = Tm, Dir = Dir, Sxx = Sxx, Sxy = Sxy, Syy = Syy)
end

"""
    total_action(m)

Σ N dσ dθ dx dy over the domain (wave action): conserved by the transport scheme
(up to boundary fluxes, dry-cell absorption and sources).
"""
function total_action(m::WaveModel)
    sp = m.spec
    s = 0.0
    for j in NG+1:NG+m.grid.ny, i in NG+1:NG+m.grid.nx, it in 1:sp.nth, is in 1:sp.ns
        s += m.N[is, it, i, j] * sp.dsig[is]
    end
    return s * sp.dth * m.grid.dx * m.grid.dy
end

end # module
