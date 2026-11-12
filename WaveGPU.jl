"""
WaveGPU.jl — GPU (and portable CPU) execution of the WaveAction / WaveSweep wave models with KernelAbstractions.jl.

The same kernels run on any KernelAbstractions backend: `CUDA.CUDABackend()`, `AMDGPU.ROCBackend()`, `Metal.MetalBackend()`,
`oneAPI.oneAPIBackend()` or `CPU()` (used for testing here: the sandbox this was written in has no GPU, so the kernels have
been validated against the CPU solvers on the CPU backend only — see test_gpu.jl).

    using CUDA                                   # or AMDGPU / Metal / oneAPI
    include("WaveAction.jl"); include("WaveSweep.jl"); include("WaveGPU.jl")
    using .WaveAction, .WaveSweep, .WaveGPU

    m  = WaveModel(...)                           # set up exactly as for the CPU (depth, spectrum, physics, boundaries, wind, GSE)
    gm = gpu_model(m; backend = CUDABackend(), T = Float32)      # uploads everything once (T = Float64 is the default)

    run_gpu!(gm; tend = 3600.0)                   # explicit solver (WaveAction.run!)
    run_implicit_gpu!(gm; tend = 3600.0, dt = 60) # implicit four-sweep solver (WaveSweep.run_implicit!)
    solve_stationary_gpu!(gm)                     # stationary four-sweep solver
    b = bulk_gpu(gm)                              # (Hs, Tm, Dir) as host matrices
    to_cpu!(gm)                                   # copy N and t back into the CPU model `m` (for output / BinWaves / plotting)

What is on the device
  * all state and environment arrays; boundary conditions (open / zero / imposed spectrum, incl. time-varying scale);
  * explicit solver: limited MUSCL flux divergence + θ/σ transport, wind, whitecapping, breaking, friction, GSE (explicit sub-steps);
  * implicit solver: the four-sweep Gauss–Seidel scheme, one launch per anti-diagonal wavefront, one thread per (cell, frequency)
    (per cell when current/depth-rate frequency shifting is present, to keep the σ Gauss–Seidel order of the CPU code);
  * stationary solver with the SWAN-like convergence test evaluated on the device.
Not on the device (use the CPU model): the steady GSE diffusion of `solve_stationary!` (paused anyway), the ADI branch of the
GSE filter (the GPU always uses explicit sub-stepping, which is the more accurate of the two), radiation stresses in `bulk`,
and `update_environment!` (change depth/currents on the CPU model, then call `refresh_environment!(gm)`).

Everything is generic in the floating-point type `T` (Float32 is typically 2–30× faster on GPUs, see the notes in README_GPU.md).
"""
module WaveGPU

using KernelAbstractions
import Adapt
using Printf
using Main.WaveAction
using Main.WaveSweep

export GPUModel, gpu_model, to_cpu!, upload_state!, refresh_environment!, refresh_boundaries!,
       step_gpu!, run_gpu!, step_implicit_gpu!, run_implicit_gpu!, solve_stationary_gpu!, bulk_gpu, hs_tm_gpu, total_action_gpu

const NG = WaveAction.NG
const minmod3 = WaveAction.minmod3
const limslope = WaveAction.limslope

# ------------------------------------------------------------------------------------------------------------
# Device data (everything a kernel needs, in one Adapt-able struct)
# ------------------------------------------------------------------------------------------------------------
struct DPar{T}
    ns::Int; nth::Int; nx::Int; ny::Int
    idx::T; idy::T; dth::T; idth::T
    g::T; hmin::T; θl::T; o2::T
    refraction::Bool; wind::Bool; whitecap::Bool; breaking::Bool; friction::Bool; hasshift::Bool
    Cds::T; delta_ds::T; p_ds::T; stpm::T; alpha_bj::T; gamma_bj::T; Cbf::T; Cl::T; alpha_pm::T; rhoratio::T
    bcl::Int; bcr::Int; bcb::Int; bct::Int          # 0 open (zero-gradient), 1 imposed spectrum, 2 zero
end

struct Dev{T,V<:AbstractVector{T},M<:AbstractMatrix{T},BM<:AbstractMatrix{Bool},A3<:AbstractArray{T,3},A4<:AbstractArray{T,4}}
    par::DPar{T}
    sig::V; sigf::V; dsig::V; cth::V; sth::V; rdp::V; rdc::V
    d::M; u10x::M; u10y::M; Ufx::M; Vfy::M
    wet::BM; refr::BM; wetx::BM; wety::BM
    cgx::A3; cgy::A3; ssh::A3; kk::A3; sh2i::A3; dec::A3; gdp::A3; gdm::A3; gUp::A3; gUm::A3; bcv::A3
    csig::A4; N::A4; N1::A4; dN::A4
    W::A3; act::BM                                   # sweep scratch: (nwork, nth, 9) and (nwork, nth)
    Dsx::A3; Dnx::A3; Dsy::A3; Dny::A3; gwx::BM; gwy::BM     # GSE face diffusivities
end
Adapt.@adapt_structure Dev

mutable struct GPUModel{T,B,DV}
    m::WaveModel
    backend::B
    D::DV
    t::Float64
    dtmin::Float64
    Hs::Any; Tm::Any; Hso::Any; Tmo::Any; wetint::Any
    bcstage::Array{T,3}
    nwet::Int
    gse_cache::Any
end
Base.show(io::IO, g::GPUModel{T}) where {T} = print(io, "GPUModel{$T} on ", typeof(g.backend), ": grid ", g.m.grid.nx, "×", g.m.grid.ny,
    ", spectrum ", g.m.spec.ns, "×", g.m.spec.nth, ", t = ", g.t)

todev(backend, T, A::AbstractArray) = (B = KernelAbstractions.allocate(backend, T, size(A)); copyto!(B, T.(A)); B)
todevB(backend, A::AbstractArray{Bool}) = (B = KernelAbstractions.allocate(backend, Bool, size(A)); copyto!(B, A); B)
zerosdev(backend, T, dims...) = (B = KernelAbstractions.allocate(backend, T, dims...); fill!(B, zero(T)); B)

bckind(b) = b isa SpectrumBC ? 1 : (b isa ZeroBC ? 2 : 0)

function make_par(m::WaveModel, ::Type{T}) where {T}
    p, sp, gr, e = m.params, m.spec, m.grid, m.env
    DPar{T}(sp.ns, sp.nth, gr.nx, gr.ny, T(1 / gr.dx), T(1 / gr.dy), T(sp.dth), T(1 / sp.dth),
            T(p.g), T(p.hmin), T(p.theta), T(p.order > 1 ? 1 : 0),
            p.refraction, p.wind, p.whitecap, p.breaking, p.friction, e.has_shift,
            T(p.Cds), T(p.delta_ds), T(p.p_ds), T(p.stpm), T(p.alpha_bj), T(p.gamma_bj), T(p.Cbf), T(p.Cl), T(p.alpha_pm), T(p.rhoratio),
            bckind(m.bc[1]), bckind(m.bc[2]), bckind(m.bc[3]), bckind(m.bc[4]))
end

function build_dev(m::WaveModel, backend, ::Type{T}; N = nothing) where {T}
    sp, e, gr = m.spec, m.env, m.grid
    ns, nth = sp.ns, sp.nth
    nxg, nyg = gr.nx + 2NG, gr.ny + 2NG
    nwork = ns * min(gr.nx, gr.ny)
    dv(A) = todev(backend, T, A)
    Nd = N === nothing ? todev(backend, T, m.N) : N
    Dev(make_par(m, T),
        dv(sp.sig), dv(sp.sigf), dv(sp.dsig), dv(sp.cth), dv(sp.sth), dv(sp.rdp), dv(sp.rdc),
        dv(e.d), dv(e.u10x), dv(e.u10y), dv(e.Ufx), dv(e.Vfy),
        todevB(backend, e.wet), todevB(backend, e.refr), todevB(backend, e.wetx), todevB(backend, e.wety),
        dv(e.cgx), dv(e.cgy), dv(e.ssh), dv(e.kk), dv(e.sh2i), zerosdev(backend, T, ns, nxg, nyg),
        dv(e.gdp), dv(e.gdm), dv(e.gUp), dv(e.gUm), zerosdev(backend, T, ns, nth, 4),
        dv(e.csig), Nd, zerosdev(backend, T, ns, nth, nxg, nyg), zerosdev(backend, T, ns, nth, nxg, nyg),
        zerosdev(backend, T, nwork, nth, 9), todevB(backend, fill(false, nwork, nth)),
        zerosdev(backend, T, ns, nxg, nyg), zerosdev(backend, T, ns, nxg, nyg), zerosdev(backend, T, ns, nxg, nyg), zerosdev(backend, T, ns, nxg, nyg),
        todevB(backend, fill(false, nxg, nyg)), todevB(backend, fill(false, nxg, nyg)))
end

"""
    gpu_model(m::WaveModel; backend = CPU(), T = Float64) -> GPUModel

Copy a configured CPU model to the device. Boundary conditions, wind and GSE settings are those of `m` at call time
(refresh with `refresh_boundaries!` / `refresh_environment!` after changing them).
"""
function gpu_model(m::WaveModel; backend = CPU(), T::Type = Float64)
    D = build_dev(m, backend, T)
    nx, ny = m.grid.nx, m.grid.ny
    wi = m.env.wet[NG+1:NG+nx, NG+1:NG+ny]
    gm = GPUModel{T,typeof(backend),typeof(D)}(m, backend, D, m.t,
        minimum(@view m.env.dtcell[NG+1:NG+nx, NG+1:NG+ny]),
        zerosdev(backend, T, nx, ny), zerosdev(backend, T, nx, ny), zerosdev(backend, T, nx, ny), zerosdev(backend, T, nx, ny),
        todevB(backend, wi), zeros(T, m.spec.ns, m.spec.nth, 4), count(wi), nothing)
    m.gse === nothing || (m.gsec === nothing && WaveAction.gse_prepare!(m); upload_gse!(gm))
    return gm
end

"Re-read depth, currents, wind and boundary types from the CPU model (after `set_environment!`, `set_wind!`, `set_boundaries!`)."
function refresh_environment!(gm::GPUModel{T}) where {T}
    m = gm.m
    gm.D = build_dev(m, gm.backend, T; N = gm.D.N)
    nx, ny = m.grid.nx, m.grid.ny
    gm.dtmin = minimum(@view m.env.dtcell[NG+1:NG+nx, NG+1:NG+ny])
    wi = m.env.wet[NG+1:NG+nx, NG+1:NG+ny]
    gm.wetint = todevB(gm.backend, wi); gm.nwet = count(wi)
    gm.gse_cache = nothing
    m.gse === nothing || (m.gsec === nothing && WaveAction.gse_prepare!(m); upload_gse!(gm))
    return gm
end
refresh_boundaries!(gm::GPUModel) = (gm.D = withpar(gm.D, make_par(gm.m, eltype(gm.D.N))); gm)

function withpar(D::Dev, par)
    vals = [getfield(D, k) for k in fieldnames(Dev) if k !== :par]
    return Dev(par, vals...)
end

function upload_gse!(gm::GPUModel{T}) where {T}
    c = gm.m.gsec
    gm.gse_cache === c && return gm
    D = gm.D; b = gm.backend
    copyto!(D.Dsx, T.(c.Dsx)); copyto!(D.Dnx, T.(c.Dnx)); copyto!(D.Dsy, T.(c.Dsy)); copyto!(D.Dny, T.(c.Dny))
    copyto!(D.gwx, c.gwx); copyto!(D.gwy, c.gwy)
    gm.gse_cache = c
    return gm
end

"Copy the device state into the CPU model (N, time)."
function to_cpu!(gm::GPUModel)
    gm.m.N .= Float64.(Array(gm.D.N)); gm.m.t = gm.t
    return gm.m
end
"Copy the CPU model's state (N, time) to the device, e.g. after `initialize!`."
function upload_state!(gm::GPUModel{T}) where {T}
    copyto!(gm.D.N, T.(gm.m.N)); gm.t = gm.m.t
    return gm
end

# ------------------------------------------------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------------------------------------------------
@kernel function k_bcx!(N, D)
    is, it, j, gg = @index(Global, NTuple)
    p = D.par; nxg = p.nx + 2NG
    side = gg > NG ? 2 : 1
    g = side == 1 ? gg : gg - NG
    ig = side == 1 ? g : nxg - g + 1
    kind = side == 1 ? p.bcl : p.bcr
    @inbounds if kind == 1
        N[is, it, ig, j] = D.bcv[is, it, side]
    elseif kind == 2
        N[is, it, ig, j] = zero(eltype(N))
    else
        N[is, it, ig, j] = N[is, it, side == 1 ? NG + 1 : nxg - NG, j]
    end
end
@kernel function k_bcy!(N, D)
    is, it, i, gg = @index(Global, NTuple)
    p = D.par; nyg = p.ny + 2NG
    side = gg > NG ? 2 : 1
    g = side == 1 ? gg : gg - NG
    jg = side == 1 ? g : nyg - g + 1
    kind = side == 1 ? p.bcb : p.bct
    @inbounds if kind == 1
        N[is, it, i, jg] = D.bcv[is, it, side+2]
    elseif kind == 2
        N[is, it, i, jg] = zero(eltype(N))
    else
        N[is, it, i, jg] = N[is, it, i, side == 1 ? NG + 1 : nyg - NG]
    end
end

function apply_bc_gpu!(gm::GPUModel{T}, N, t) where {T}
    D, m = gm.D, gm.m
    p = D.par
    anyspec = false
    for k in 1:4
        b = m.bc[k]
        if b isa SpectrumBC
            s = b.scale(t)
            @views gm.bcstage[:, :, k] .= T.((s * s) .* m.Nbc[k])
            anyspec = true
        end
    end
    anyspec && copyto!(D.bcv, gm.bcstage)
    be = gm.backend
    k_bcx!(be)(N, D; ndrange = (p.ns, p.nth, p.ny + 2NG, 2NG))
    k_bcy!(be)(N, D; ndrange = (p.ns, p.nth, p.nx + 2NG, 2NG))
    return N
end

# ------------------------------------------------------------------------------------------------------------
# Explicit solver: tendency, advance, sources
# ------------------------------------------------------------------------------------------------------------
@inline function xflux(D, N, is, it, i, j, c0)
    T = eltype(N); p = D.par
    @inbounds if D.wetx[i, j]
        cx = D.cgx[is, i, j] * c0 + D.Ufx[i, j]
        Nl = N[is, it, i-1, j]; Nr = N[is, it, i, j]
        sl = p.o2 * limslope(N[is, it, i-2, j], Nl, Nr, p.θl)
        sr = p.o2 * limslope(Nl, Nr, N[is, it, i+1, j], p.θl)
        return ifelse(cx > 0, cx * (Nl + T(0.5) * sl), cx * (Nr - T(0.5) * sr))
    else
        return zero(T)
    end
end
@inline function yflux(D, N, is, it, i, j, s0)
    T = eltype(N); p = D.par
    @inbounds if D.wety[i, j]
        cy = D.cgy[is, i, j] * s0 + D.Vfy[i, j]
        Nb = N[is, it, i, j-1]; Nt = N[is, it, i, j]
        sb = p.o2 * limslope(N[is, it, i, j-2], Nb, Nt, p.θl)
        st = p.o2 * limslope(Nb, Nt, N[is, it, i, j+1], p.θl)
        return ifelse(cy > 0, cy * (Nb + T(0.5) * sb), cy * (Nt - T(0.5) * st))
    else
        return zero(T)
    end
end

@kernel function k_tendency!(dN, N, D)
    is, it, ii, jj = @index(Global, NTuple)
    i = ii + NG; j = jj + NG
    T = eltype(N); p = D.par
    ns, nth = p.ns, p.nth
    @inbounds begin
        if !D.wet[i, j]
            dN[is, it, i, j] = zero(T)
        else
            c0, s0 = D.cth[it], D.sth[it]
            div = (xflux(D, N, is, it, i + 1, j, c0) - xflux(D, N, is, it, i, j, c0)) * p.idx +
                  (yflux(D, N, is, it, i, j + 1, s0) - yflux(D, N, is, it, i, j, s0)) * p.idy
            dorefr = D.refr[i, j]
            val = -div
            if dorefr
                itp = it == nth ? 1 : it + 1; itm = it == 1 ? nth : it - 1
                itpp = itp == nth ? 1 : itp + 1; itmm = itm == 1 ? nth : itm - 1
                Nc = N[is, it, i, j]
                ssh = D.ssh[is, i, j]
                Np = N[is, itp, i, j]; Nm = N[is, itm, i, j]
                sc = p.o2 * limslope(Nm, Nc, Np, p.θl)
                spl = p.o2 * limslope(Nc, Np, N[is, itpp, i, j], p.θl)
                sm = p.o2 * limslope(N[is, itmm, i, j], Nm, Nc, p.θl)
                cthp = ssh * D.gdp[it, i, j] + D.gUp[it, i, j]
                cthm = ssh * D.gdm[it, i, j] + D.gUm[it, i, j]
                Fp = ifelse(cthp > 0, cthp * (Nc + T(0.5) * sc), cthp * (Np - T(0.5) * spl))
                Fm = ifelse(cthm > 0, cthm * (Nm + T(0.5) * sm), cthm * (Nc - T(0.5) * sc))
                val = -(div + (Fp - Fm) * p.idth)
                if p.hasshift
                    σ = D.sig[is]
                    csp = D.csig[1, is+1, i, j] + D.csig[2, is+1, i, j] * c0 * c0 + D.csig[3, is+1, i, j] * c0 * s0 + D.csig[4, is+1, i, j] * s0 * s0
                    csm = D.csig[1, is, i, j] + D.csig[2, is, i, j] * c0 * c0 + D.csig[3, is, i, j] * c0 * s0 + D.csig[4, is, i, j] * s0 * s0
                    slc = zero(T)
                    if p.o2 > 0 && is > 1 && is < ns
                        slc = minmod3(p.θl * (Nc - N[is-1, it, i, j]) * D.rdp[is-1], (N[is+1, it, i, j] - N[is-1, it, i, j]) * D.rdc[is],
                                      p.θl * (N[is+1, it, i, j] - Nc) * D.rdp[is])
                    end
                    Fps = zero(T)
                    if is == ns
                        Fps = csp > 0 ? csp * Nc : zero(T)
                    else
                        Nn = N[is+1, it, i, j]
                        sln = zero(T)
                        if p.o2 > 0 && is + 1 < ns
                            sln = minmod3(p.θl * (Nn - Nc) * D.rdp[is], (N[is+2, it, i, j] - Nc) * D.rdc[is+1],
                                          p.θl * (N[is+2, it, i, j] - Nn) * D.rdp[is+1])
                        end
                        Fps = csp > 0 ? csp * (Nc + slc * (D.sigf[is+1] - σ)) : csp * (Nn + sln * (D.sigf[is+1] - D.sig[is+1]))
                    end
                    Fms = zero(T)
                    if is > 1
                        Nl = N[is-1, it, i, j]
                        sll = zero(T)
                        if p.o2 > 0 && is - 1 > 1
                            sll = minmod3(p.θl * (Nl - N[is-2, it, i, j]) * D.rdp[is-2], (Nc - N[is-2, it, i, j]) * D.rdc[is-1],
                                          p.θl * (Nc - Nl) * D.rdp[is-1])
                        end
                        Fms = csm > 0 ? csm * (Nl + sll * (D.sigf[is] - D.sig[is-1])) : csm * (Nc + slc * (D.sigf[is] - σ))
                    end
                    val -= (Fps - Fms) / D.dsig[is]
                end
            end
            dN[is, it, i, j] = val
        end
    end
end

@kernel function k_advance!(out, N0, dN, D, dt)
    is, it, ii, jj = @index(Global, NTuple)
    i = ii + NG; j = jj + NG
    @inbounds out[is, it, i, j] = D.wet[i, j] ? max(N0[is, it, i, j] + dt * dN[is, it, i, j], zero(eltype(out))) : zero(eltype(out))
end

# ---- source terms -------------------------------------------------------------------------------------------
@inline function bfrac(b::T) where {T}
    b >= 1 && return one(T)
    b < T(0.2) && return zero(T)
    lo = T(1e-12); hi = one(T) - T(1e-7); b2 = b * b
    for _ in 1:40
        mid = T(0.5) * (lo + hi)
        if 1 - mid + b2 * log(mid) < 0
            lo = mid
        else
            hi = mid
        end
    end
    return T(0.5) * (lo + hi)
end

# bulk spectral statistics and the frequency-independent dissipation factors of one cell
@inline function cellstats(D, N, i, j)
    T = eltype(N); p = D.par
    Etot = zero(T); m1 = zero(T); mm1 = zero(T); sk = zero(T)
    @inbounds for is in 1:p.ns
        σ = D.sig[is]
        s = zero(T)
        for it in 1:p.nth
            s += N[is, it, i, j]
        end
        Es = σ * s * p.dth * D.dsig[is]
        Etot += Es; m1 += σ * Es; mm1 += Es / σ; sk += sqrt(D.kk[is, i, j]) * Es
    end
    haveE = Etot > T(1e-12)
    sbar = haveE ? m1 / Etot : one(T)
    stil = haveE ? Etot / mm1 : one(T)
    ktil = haveE ? (sk / Etot)^2 : one(T)
    steep = ktil * sqrt(Etot)
    brate = zero(T)
    if p.breaking && haveE
        Hm = p.gamma_bj * D.d[i, j]
        Qb = bfrac(sqrt(8Etot) / Hm)
        brate = T(0.25) * p.alpha_bj * Qb * (sbar / T(2π)) * Hm^2 / Etot
    end
    wcfac = (p.whitecap && haveE) ? p.Cds * (steep / sqrt(p.stpm))^p.p_ds * stil : zero(T)
    return haveE, brate, wcfac, ktil
end
@inline function decbin(D, is, i, j, haveE, brate, wcfac, ktil)
    T = typeof(brate); p = D.par
    σ = D.sig[is]; k = D.kk[is, i, j]
    dec = brate
    p.whitecap && haveE && (dec += wcfac * ((1 - p.delta_ds) + p.delta_ds * k / ktil) * (k / ktil))
    p.friction && (dec += p.Cbf * σ^2 / p.g^2 * D.sh2i[is, i, j])
    return dec
end

@kernel function k_sources!(N, D, dt)
    ii, jj = @index(Global, NTuple)
    i = ii + NG; j = jj + NG
    T = eltype(N); p = D.par
    @inbounds if D.wet[i, j]
        haveE, brate, wcfac, ktil = cellstats(D, N, i, j)
        u10x, u10y = D.u10x[i, j], D.u10y[i, j]
        U10 = hypot(u10x, u10y)
        dowind = p.wind && U10 > T(0.1)
        ustar = zero(T); cw = one(T); sw = zero(T); σPM = one(T)
        if dowind
            Cd = U10 < T(7.5) ? T(1.2875e-3) : (T(0.8) + T(0.065) * U10) * T(1e-3)
            ustar = U10 * sqrt(Cd)
            cw, sw = u10x / U10, u10y / U10
            σPM = T(2π) * T(0.13) * p.g / (28 * ustar)
        end
        for is in 1:p.ns
            σ = D.sig[is]; k = D.kk[is, i, j]
            dec = decbin(D, is, i, j, haveE, brate, wcfac, ktil)
            cph = σ / k
            Alin = zero(T); dNmax = zero(T)
            if dowind
                σ2 = σ * σ
                Alin = T(1.5e-3) / (T(2π) * p.g^2) * exp(-(σ / σPM)^(-4))
                dNmax = p.Cl * p.alpha_pm * p.g^2 / (T(2π) * σ2 * σ2 * σ2)
            end
            for it in 1:p.nth
                Nv = N[is, it, i, j]
                dNw = zero(T)
                if dowind
                    cosd = D.cth[it] * cw + D.sth[it] * sw
                    uc = ustar * max(zero(T), cosd)
                    A = Alin * uc^4
                    B = max(zero(T), T(0.25) * p.rhoratio * (28 * ustar / cph * cosd - 1)) * σ
                    dNw = min(dt * (A / σ + B * Nv), dNmax)
                end
                N[is, it, i, j] = (Nv + dNw) / (1 + dt * dec)
            end
        end
    end
end

@kernel function k_dissipation!(D, N)
    ii, jj = @index(Global, NTuple)
    i = ii + NG; j = jj + NG
    @inbounds if D.wet[i, j]
        haveE, brate, wcfac, ktil = cellstats(D, N, i, j)
        for is in 1:D.par.ns
            D.dec[is, i, j] = decbin(D, is, i, j, haveE, brate, wcfac, ktil)
        end
    end
end

# ---- garden sprinkler effect: one explicit sub-step (out-of-place, reads N only) ---------------------------------
@inline function gfx(D, N, is, it, i, j, sc, cc, ss, cs, idx, idy)
    T = eltype(N)
    @inbounds if D.gwx[i, j]
        Dss = D.Dsx[is, i, j] * sc; Dnn = D.Dnx[is, i, j] * sc
        gy = T(0.25) * idy * ((N[is, it, i-1, j+1] - N[is, it, i-1, j-1]) + (N[is, it, i, j+1] - N[is, it, i, j-1]))
        return -((Dss * cc + Dnn * ss) * (N[is, it, i, j] - N[is, it, i-1, j]) * idx + (Dss - Dnn) * cs * gy)
    else
        return zero(T)
    end
end
@inline function gfy(D, N, is, it, i, j, sc, cc, ss, cs, idx, idy)
    T = eltype(N)
    @inbounds if D.gwy[i, j]
        Dss = D.Dsy[is, i, j] * sc; Dnn = D.Dny[is, i, j] * sc
        gx = T(0.25) * idx * ((N[is, it, i+1, j-1] - N[is, it, i-1, j-1]) + (N[is, it, i+1, j] - N[is, it, i-1, j]))
        return -((Dss - Dnn) * cs * gx + (Dss * ss + Dnn * cc) * (N[is, it, i, j] - N[is, it, i, j-1]) * idy)
    else
        return zero(T)
    end
end
@kernel function k_gse!(out, N, D, dτ, sc, clip)
    is, it, ii, jj = @index(Global, NTuple)
    i = ii + NG; j = jj + NG
    p = D.par
    @inbounds if D.wet[i, j]
        c0, s0 = D.cth[it], D.sth[it]
        cc, ss, cs = c0 * c0, s0 * s0, c0 * s0
        div = (gfx(D, N, is, it, i + 1, j, sc, cc, ss, cs, p.idx, p.idy) - gfx(D, N, is, it, i, j, sc, cc, ss, cs, p.idx, p.idy)) * p.idx +
              (gfy(D, N, is, it, i, j + 1, sc, cc, ss, cs, p.idx, p.idy) - gfy(D, N, is, it, i, j, sc, cc, ss, cs, p.idx, p.idy)) * p.idy
        v = N[is, it, i, j] - dτ * div
        out[is, it, i, j] = clip ? max(v, zero(eltype(N))) : v
    end
end
@kernel function k_copyint!(dst, src, D)
    is, it, ii, jj = @index(Global, NTuple)
    i = ii + NG; j = jj + NG
    @inbounds D.wet[i, j] && (dst[is, it, i, j] = src[is, it, i, j])
end

function gse_step_gpu!(gm::GPUModel{T}, dt) where {T}
    m = gm.m; g = m.gse
    g === nothing && return 0
    m.gsec === nothing && WaveAction.gse_prepare!(m)
    upload_gse!(gm)
    c = m.gsec; D = gm.D; p = D.par
    sc = c.dtmode ? T(dt) : one(T)
    dxm = min(m.grid.dx, m.grid.dy)
    nsub = max(1, ceil(Int, g.safety * dt * c.Dsum * (c.dtmode ? dt : 1.0) / dxm^2))
    dτ = T(dt / nsub)
    be = gm.backend
    for k in 1:nsub
        apply_bc_gpu!(gm, D.N, gm.t)
        k_gse!(be)(D.dN, D.N, D, dτ, sc, k == nsub; ndrange = (p.ns, p.nth, p.nx, p.ny))
        k_copyint!(be)(D.N, D.dN, D; ndrange = (p.ns, p.nth, p.nx, p.ny))
    end
    return nsub
end

# ------------------------------------------------------------------------------------------------------------
# Explicit time stepping
# ------------------------------------------------------------------------------------------------------------
function tendency_gpu!(gm::GPUModel, dN, N)
    p = gm.D.par
    k_tendency!(gm.backend)(dN, N, gm.D; ndrange = (p.ns, p.nth, p.nx, p.ny))
end
function advance_gpu!(gm::GPUModel{T}, out, N0, dN, dt) where {T}
    p = gm.D.par
    k_advance!(gm.backend)(out, N0, dN, gm.D, T(dt); ndrange = (p.ns, p.nth, p.nx, p.ny))
end
function sources_gpu!(gm::GPUModel{T}, dt) where {T}
    p = gm.D.par
    (p.wind || p.whitecap || p.breaking || p.friction) || return
    k_sources!(gm.backend)(gm.D.N, gm.D, T(dt); ndrange = (p.nx, p.ny))
end

"One predictor/corrector step on the device (see `WaveAction.step!`)."
function step_gpu!(gm::GPUModel, dt)
    D = gm.D
    apply_bc_gpu!(gm, D.N, gm.t)
    tendency_gpu!(gm, D.dN, D.N)
    advance_gpu!(gm, D.N1, D.N, D.dN, 0.5dt)
    apply_bc_gpu!(gm, D.N1, gm.t + 0.5dt)
    tendency_gpu!(gm, D.dN, D.N1)
    advance_gpu!(gm, D.N, D.N, D.dN, dt)
    sources_gpu!(gm, dt)
    gm.t += dt
    gse_step_gpu!(gm, dt)
    return gm
end

stable_dt_gpu(gm::GPUModel, cfl = 0.4) = min(cfl, 0.5) * gm.dtmin

function finite_check(gm::GPUModel)
    all(isfinite, gm.D.N) || error("non-finite wave action at t = $(gm.t) s")
end

"""
    run_gpu!(gm; tend, cfl = 0.4, dt = nothing, output_dt = nothing, callback = nothing, verbose = true)

Explicit solver on the device, same semantics as `WaveAction.run!`; `callback(gm, t)` is called at the start and at each output time
(use `bulk_gpu(gm)` or `to_cpu!(gm)` inside it). The finite-value check runs every 20 steps.
"""
function run_gpu!(gm::GPUModel; tend, cfl = 0.4, dt = nothing, output_dt = nothing, callback = nothing, verbose = true)
    dtmax = dt === nothing ? stable_dt_gpu(gm, cfl) : dt
    isfinite(dtmax) || error("no wet cells: nothing to compute")
    verbose && println("WaveGPU (explicit): dt = $(round(dtmax, digits = 4)) s, $(typeof(gm.backend)), grid $(gm.m.grid.nx)×$(gm.m.grid.ny), spectrum $(gm.m.spec.ns)×$(gm.m.spec.nth)")
    nextout = output_dt === nothing ? Inf : gm.t + output_dt
    callback === nothing || callback(gm, gm.t)
    nsteps = 0; tol = 1e-9 * max(1.0, abs(tend))
    while gm.t < tend - tol
        h = min(dtmax, tend - gm.t, nextout - gm.t)
        step_gpu!(gm, h)
        nsteps += 1
        if gm.t >= nextout - tol
            callback === nothing || callback(gm, gm.t)
            nextout += output_dt
        end
        nsteps % 20 == 0 && finite_check(gm)
    end
    finite_check(gm)
    KernelAbstractions.synchronize(gm.backend)
    return nsteps
end

# ------------------------------------------------------------------------------------------------------------
# Implicit four-sweep solver
# ------------------------------------------------------------------------------------------------------------
@inline Hs_(c) = c > 0 ? one(c) : zero(c)

# Thomas algorithm on the segment lo:hi of the scratch columns (1 a, 2 b, 3 c, 4 r, 5 x, 8 gam)
@inline function tridiag_w!(W, wid, lo, hi)
    @inbounds begin
        bet = W[wid, lo, 2]
        W[wid, lo, 5] = W[wid, lo, 4] / bet
        for q in lo+1:hi
            W[wid, q, 8] = W[wid, q-1, 3] / bet
            bet = W[wid, q, 2] - W[wid, q, 1] * W[wid, q, 8]
            W[wid, q, 5] = (W[wid, q, 4] - W[wid, q, 1] * W[wid, q-1, 5]) / bet
        end
        for q in hi-1:-1:lo
            W[wid, q, 5] -= W[wid, q+1, 8] * W[wid, q+1, 5]
        end
    end
end
# the same with right-hand side column `rc` and result column `xc` (for the Sherman–Morrison solves; diagonal in column 7)
@inline function tridiag_cols!(W, wid, n, dcol, rcol, xcol)
    @inbounds begin
        bet = W[wid, 1, dcol]
        W[wid, 1, xcol] = W[wid, 1, rcol] / bet
        for q in 2:n
            W[wid, q, 8] = W[wid, q-1, 3] / bet
            bet = W[wid, q, dcol] - W[wid, q, 1] * W[wid, q, 8]
            W[wid, q, xcol] = (W[wid, q, rcol] - W[wid, q, 1] * W[wid, q-1, xcol]) / bet
        end
        for q in n-1:-1:1
            W[wid, q, xcol] -= W[wid, q+1, 8] * W[wid, q+1, xcol]
        end
    end
end
@inline function cyclic_w!(W, wid, n)
    T = eltype(W)
    @inbounds begin
        alpha = W[wid, n, 3]; beta = W[wid, 1, 1]
        gamma = -W[wid, 1, 2]
        for k in 1:n
            W[wid, k, 7] = W[wid, k, 2]; W[wid, k, 9] = zero(T)
        end
        W[wid, 1, 7] = W[wid, 1, 2] - gamma
        W[wid, n, 7] = W[wid, n, 2] - alpha * beta / gamma
        W[wid, 1, 9] = gamma; W[wid, n, 9] = alpha
        tridiag_cols!(W, wid, n, 7, 4, 5)       # x
        tridiag_cols!(W, wid, n, 7, 9, 6)       # z
        fact = (W[wid, 1, 5] + beta * W[wid, n, 5] / gamma) / (1 + W[wid, 1, 6] + beta * W[wid, n, 6] / gamma)
        for k in 1:n
            W[wid, k, 5] -= fact * W[wid, k, 6]
        end
    end
end

@inline function sweep_bin!(D, wid, i, j, is, sx, sy, dtinv, Nn, wθ, wσ)
    T = eltype(D.N); p = D.par; N = D.N; W = D.W; act = D.act
    ns, nth = p.ns, p.nth
    idx, idy, dth = p.idx, p.idy, p.dth
    dorefr = D.refr[i, j]
    doshift = p.hasshift && dorefr
    dissip = p.whitecap || p.breaking || p.friction
    g = p.g
    Ufl, Ufr, Vfb, Vft = D.Ufx[i, j], D.Ufx[i+1, j], D.Vfy[i, j], D.Vfy[i, j+1]
    gl = i == NG + 1 && p.bcl == 0
    gr_ = i == NG + p.nx && p.bcr == 0
    gb = j == NG + 1 && p.bcb == 0
    gt = j == NG + p.ny && p.bct == 0
    u10x, u10y = D.u10x[i, j], D.u10y[i, j]
    U10 = hypot(u10x, u10y)
    dowind = p.wind && U10 > T(0.1)
    ustar = zero(T); cw = one(T); sw_ = zero(T); σPM = one(T)
    if dowind
        Cd = U10 < T(7.5) ? T(1.2875e-3) : (T(0.8) + T(0.065) * U10) * T(1e-3)
        ustar = U10 * sqrt(Cd)
        cw, sw_ = u10x / U10, u10y / U10
        σPM = T(2π) * T(0.13) * g / (28 * ustar)
    end
    @inbounds begin
        σ = D.sig[is]
        ssh = D.ssh[is, i, j]
        cgxl, cgxr, cgyb, cgyt = D.cgx[is, i, j], D.cgx[is, i+1, j], D.cgy[is, i, j], D.cgy[is, i, j+1]
        dσ = D.dsig[is]
        nact = 0; t0 = 0
        for it in 1:nth
            c0, s0 = D.cth[it], D.sth[it]
            isact = (((cgxl + cgxr) * c0 + Ufl + Ufr) >= 0 ? 1 : -1) == sx && (((cgyb + cgyt) * s0 + Vfb + Vft) >= 0 ? 1 : -1) == sy
            act[wid, it] = isact
            if isact; nact += 1; else t0 = it; end
        end
        nact == 0 && return nothing
        ring = nact == nth
        ring && (t0 = 0)
        decis = dissip ? D.dec[is, i, j] : zero(T)
        Alin = zero(T); dNmax = T(Inf); cph = one(T)
        if dowind
            σ2 = σ * σ
            Alin = T(1.5e-3) / (T(2π) * g^2) * exp(-(σ / σPM)^(-4))
            dNmax = p.Cl * p.alpha_pm * g^2 / (T(2π) * σ2 * σ2 * σ2)
            cph = σ / D.kk[is, i, j]
        end
        for q in 1:nth
            it = t0 + q; it > nth && (it -= nth)
            W[wid, q, 1] = zero(T); W[wid, q, 3] = zero(T)
            act[wid, it] || continue
            c0, s0 = D.cth[it], D.sth[it]
            cxl = cgxl * c0 + Ufl; cxr = cgxr * c0 + Ufr
            cyb = cgyb * s0 + Vfb; cyt = cgyt * s0 + Vft
            Nc = N[is, it, i, j]
            diag = decis
            rhs = zero(T)
            if dowind
                cosd = c0 * cw + s0 * sw_
                uc = ustar * max(zero(T), cosd)
                rhs += Alin * uc * uc * uc * uc / σ + max(zero(T), T(0.25) * p.rhoratio * (28 * ustar / cph * cosd - 1)) * σ * Nc
            end
            if dtinv > 0
                diag += dtinv
                rhs += Nn[is, it, i, j] * dtinv
            end
            if cxl > 0 && !gl; rhs += cxl * N[is, it, i-1, j] * idx; else diag -= cxl * idx; end
            if cxr > 0 || gr_; diag += cxr * idx; else rhs -= cxr * N[is, it, i+1, j] * idx; end
            if cyb > 0 && !gb; rhs += cyb * N[is, it, i, j-1] * idy; else diag -= cyb * idy; end
            if cyt > 0 || gt; diag += cyt * idy; else rhs -= cyt * N[is, it, i, j+1] * idy; end
            if dorefr
                cthp = ssh * D.gdp[it, i, j] + D.gUp[it, i, j]
                cthm = ssh * D.gdm[it, i, j] + D.gUm[it, i, j]
                diag += (cthp * ((1 - wθ) * Hs_(cthp) + T(0.5) * wθ) - cthm * ((1 - wθ) * (1 - Hs_(cthm)) + T(0.5) * wθ)) / dth
                W[wid, q, 3] = cthp * ((1 - wθ) * (1 - Hs_(cthp)) + T(0.5) * wθ) / dth
                W[wid, q, 1] = -cthm * ((1 - wθ) * Hs_(cthm) + T(0.5) * wθ) / dth
            end
            if doshift
                csp = D.csig[1, is+1, i, j] + D.csig[2, is+1, i, j] * c0 * c0 + D.csig[3, is+1, i, j] * c0 * s0 + D.csig[4, is+1, i, j] * s0 * s0
                csm = D.csig[1, is, i, j] + D.csig[2, is, i, j] * c0 * c0 + D.csig[3, is, i, j] * c0 * s0 + D.csig[4, is, i, j] * s0 * s0
                if is == ns
                    diag += max(csp, zero(T)) / dσ
                else
                    diag += csp * ((1 - wσ) * Hs_(csp) + T(0.5) * wσ) / dσ
                    rhs -= csp * ((1 - wσ) * (1 - Hs_(csp)) + T(0.5) * wσ) * N[is+1, it, i, j] / dσ
                end
                if is > 1
                    diag -= csm * ((1 - wσ) * (1 - Hs_(csm)) + T(0.5) * wσ) / dσ
                    rhs += csm * ((1 - wσ) * Hs_(csm) + T(0.5) * wσ) * N[is-1, it, i, j] / dσ
                end
            end
            W[wid, q, 2] = max(diag, T(1e-12))
            W[wid, q, 4] = rhs
        end
        if ring
            cyclic_w!(W, wid, nth)
        else
            q = 1
            while q <= nth
                itq = t0 + q; itq > nth && (itq -= nth)
                if !act[wid, itq]
                    q += 1; continue
                end
                lo = q
                while q + 1 <= nth
                    itn = t0 + q + 1; itn > nth && (itn -= nth)
                    act[wid, itn] || break
                    q += 1
                end
                hi = q
                itlo = t0 + lo; itlo > nth && (itlo -= nth)
                ithi = t0 + hi; ithi > nth && (ithi -= nth)
                itprev = itlo == 1 ? nth : itlo - 1
                itnext = ithi == nth ? 1 : ithi + 1
                W[wid, lo, 4] -= W[wid, lo, 1] * N[is, itprev, i, j]; W[wid, lo, 1] = zero(T)
                W[wid, hi, 4] -= W[wid, hi, 3] * N[is, itnext, i, j]; W[wid, hi, 3] = zero(T)
                tridiag_w!(W, wid, lo, hi)
                q += 1
            end
        end
        for q in 1:nth
            it = t0 + q; it > nth && (it -= nth)
            act[wid, it] || continue
            v = max(W[wid, q, 5], zero(T))
            if dowind
                v = min(v, N[is, it, i, j] + dNmax)
            end
            N[is, it, i, j] = v
        end
    end
    return nothing
end

# one thread per (frequency, wavefront cell)
@kernel function k_sweep_bin!(D, Nn, sx, sy, k, ulo, dtinv, wθ, wσ)
    is, ui = @index(Global, NTuple)
    p = D.par
    u = ulo + ui - 1; v = k - u
    i = (sx > 0 ? u : p.nx + 1 - u) + NG
    j = (sy > 0 ? v : p.ny + 1 - v) + NG
    wid = is + p.ns * (ui - 1)
    @inbounds D.wet[i, j] && sweep_bin!(D, wid, i, j, is, sx, sy, dtinv, Nn, wθ, wσ)
end
# one thread per wavefront cell, frequencies in sequence (Gauss–Seidel in σ as in the CPU code; used when frequency shifting is active)
@kernel function k_sweep_cell!(D, Nn, sx, sy, k, ulo, dtinv, wθ, wσ)
    ui = @index(Global)
    p = D.par
    u = ulo + ui - 1; v = k - u
    i = (sx > 0 ? u : p.nx + 1 - u) + NG
    j = (sy > 0 ? v : p.ny + 1 - v) + NG
    @inbounds if D.wet[i, j]
        for is in 1:p.ns
            sweep_bin!(D, ui, i, j, is, sx, sy, dtinv, Nn, wθ, wσ)
        end
    end
end

function sweep_gpu!(gm::GPUModel{T}, sx, sy, dtinv, Nn, sw::SweepParams) where {T}
    D = gm.D; p = D.par; nx, ny = p.nx, p.ny; be = gm.backend
    wθ, wσ = T(sw.w_theta), T(sw.w_sigma)
    dti = T(dtinv)
    for k in 2:nx+ny
        ulo = max(1, k - ny); uhi = min(nx, k - 1)
        ulen = uhi - ulo + 1
        ulen <= 0 && continue
        if p.hasshift
            k_sweep_cell!(be)(D, Nn, sx, sy, k, ulo, dti, wθ, wσ; ndrange = ulen)
        else
            k_sweep_bin!(be)(D, Nn, sx, sy, k, ulo, dti, wθ, wσ; ndrange = (p.ns, ulen))
        end
    end
end

function update_dissipation_gpu!(gm::GPUModel)
    p = gm.D.par
    (p.whitecap || p.breaking || p.friction) || return
    k_dissipation!(gm.backend)(gm.D, gm.D.N; ndrange = (p.nx, p.ny))
end

function four_sweeps_gpu!(gm::GPUModel, dtinv, Nn, sw::SweepParams, tbc)
    for (sx, sy) in ((1, 1), (-1, 1), (-1, -1), (1, -1))
        apply_bc_gpu!(gm, gm.D.N, tbc)
        update_dissipation_gpu!(gm)
        sweep_gpu!(gm, sx, sy, dtinv, Nn, sw)
    end
    apply_bc_gpu!(gm, gm.D.N, tbc)
    return gm
end

"One backward-Euler step of the four-sweep scheme on the device (see `WaveSweep.step_implicit!`)."
function step_implicit_gpu!(gm::GPUModel, dt, sw::SweepParams = SweepParams())
    copyto!(gm.D.N1, gm.D.N)
    for _ in 1:sw.iters
        four_sweeps_gpu!(gm, 1 / dt, gm.D.N1, sw, gm.t + dt)
    end
    gm.t += dt
    gse_step_gpu!(gm, dt)
    return gm
end

"""
    run_implicit_gpu!(gm; tend, dt = 10 × explicit step, sweep = SweepParams(), output_dt = nothing, callback = nothing, verbose = true)

Same semantics as `WaveSweep.run_implicit!`.
"""
function run_implicit_gpu!(gm::GPUModel; tend, dt = nothing, sweep::SweepParams = SweepParams(), output_dt = nothing, callback = nothing, verbose = true)
    dtmax = dt === nothing ? 10 * stable_dt_gpu(gm, 0.5) : dt
    isfinite(dtmax) || error("no wet cells: nothing to compute")
    verbose && println("WaveGPU (implicit, 4 sweeps): dt = $(round(dtmax, digits = 3)) s ($(round(dtmax / stable_dt_gpu(gm, 0.5), digits = 1)) × explicit step), $(typeof(gm.backend))")
    nextout = output_dt === nothing ? Inf : gm.t + output_dt
    callback === nothing || callback(gm, gm.t)
    nsteps = 0; tol = 1e-9 * max(1.0, abs(tend))
    while gm.t < tend - tol
        h = min(dtmax, tend - gm.t, nextout - gm.t)
        step_implicit_gpu!(gm, h, sweep)
        nsteps += 1
        if gm.t >= nextout - tol
            callback === nothing || callback(gm, gm.t)
            nextout += output_dt
        end
        finite_check(gm)
    end
    KernelAbstractions.synchronize(gm.backend)
    return nsteps
end

# ------------------------------------------------------------------------------------------------------------
# Diagnostics and the stationary solver
# ------------------------------------------------------------------------------------------------------------
@kernel function k_hstm!(Hs, Tm, D)
    ii, jj = @index(Global, NTuple)
    i = ii + NG; j = jj + NG
    T = eltype(Hs); p = D.par
    m0 = zero(T); m1 = zero(T)
    @inbounds for it in 1:p.nth, is in 1:p.ns
        E = D.sig[is] * D.N[is, it, i, j] * p.dth * D.dsig[is]
        m0 += E; m1 += D.sig[is] * E
    end
    @inbounds Hs[ii, jj] = 4 * sqrt(m0)
    @inbounds Tm[ii, jj] = m1 > T(1e-20) ? T(2π) * m0 / m1 : zero(T)
end

function hs_tm_gpu!(gm::GPUModel, Hs, Tm)
    p = gm.D.par
    k_hstm!(gm.backend)(Hs, Tm, gm.D; ndrange = (p.nx, p.ny))
end
"Hs and Tm01 as host matrices (nx, ny)."
function hs_tm_gpu(gm::GPUModel)
    hs_tm_gpu!(gm, gm.Hs, gm.Tm)
    return Float64.(Array(gm.Hs)), Float64.(Array(gm.Tm))
end

@kernel function k_bulk!(Hs, Tm, Dir, D)
    ii, jj = @index(Global, NTuple)
    i = ii + NG; j = jj + NG
    T = eltype(Hs); p = D.par
    m0 = zero(T); m1 = zero(T); cc = zero(T); ss = zero(T)
    @inbounds if D.wet[i, j]
        for it in 1:p.nth, is in 1:p.ns
            E = D.sig[is] * D.N[is, it, i, j] * p.dth * D.dsig[is]
            m0 += E; m1 += D.sig[is] * E; cc += D.cth[it] * E; ss += D.sth[it] * E
        end
        Hs[ii, jj] = 4 * sqrt(m0)
        Tm[ii, jj] = m1 > T(1e-20) ? T(2π) * m0 / m1 : zero(T)
        Dir[ii, jj] = m0 > T(1e-20) ? mod(270 - atan(ss, cc) * 180 / T(π), 360) : zero(T)
    else
        Hs[ii, jj] = zero(T); Tm[ii, jj] = zero(T); Dir[ii, jj] = zero(T)
    end
end
"""
    bulk_gpu(gm) -> (Hs, Tm, Dir)

Host matrices (nx, ny); same definitions as `WaveAction.bulk` without the radiation stresses.
"""
function bulk_gpu(gm::GPUModel{T}) where {T}
    p = gm.D.par
    Dir = zerosdev(gm.backend, T, p.nx, p.ny)
    k_bulk!(gm.backend)(gm.Hs, gm.Tm, Dir, gm.D; ndrange = (p.nx, p.ny))
    return (Hs = Float64.(Array(gm.Hs)), Tm = Float64.(Array(gm.Tm)), Dir = Float64.(Array(Dir)))
end

function total_action_gpu(gm::GPUModel)
    p = gm.D.par
    s = sum(@view gm.D.N[:, :, NG+1:NG+p.nx, NG+1:NG+p.ny])
    return Float64(s) * 0 + sum(Float64, sum(@view(gm.D.N[:, :, NG+1:NG+p.nx, NG+1:NG+p.ny]); dims = (2, 3, 4)) .* reshape(gm.D.dsig, :, 1, 1, 1)) *
           Float64(gm.D.par.dth) * gm.m.grid.dx * gm.m.grid.dy
end

"""
    solve_stationary_gpu!(gm; sweep = SweepParams(), verbose = true) -> (iterations, converged)

Stationary solution by repeated four-sweep iterations with the same SWAN-like convergence test as `WaveSweep.solve_stationary!`
(the test and its reduction run on the device; only one integer is copied back per iteration). No steady GSE diffusion.
"""
function solve_stationary_gpu!(gm::GPUModel{T}; sweep::SweepParams = SweepParams(), verbose::Bool = true) where {T}
    gm.m.gse === nothing || verbose && println("  note: the steady GSE diffusion is not implemented on the device; GSE is ignored in the stationary solver")
    gm.nwet == 0 && return (0, true)
    Hs, Tm, Hso, Tmo = gm.Hs, gm.Tm, gm.Hso, gm.Tmo
    dabs, drel = T(sweep.dabs), T(sweep.drel)
    hs_tm_gpu!(gm, Hso, Tmo)
    for iter in 1:sweep.maxiter
        four_sweeps_gpu!(gm, 0.0, gm.D.N1, sweep, gm.t)
        hs_tm_gpu!(gm, Hs, Tm)
        ok = sum((abs.(Hs .- Hso) .<= max.(dabs, drel .* Hs)) .& (abs.(Tm .- Tmo) .<= max.(dabs, drel .* Tm)) .& gm.wetint)
        frac = ok / gm.nwet
        verbose && println("  iteration $iter: $(round(100frac, digits = 2)) % of wet cells converged")
        copyto!(Hso, Hs); copyto!(Tmo, Tm)
        iter > 1 && frac >= sweep.npnts && return (iter, true)
    end
    verbose && println("  stationary solver reached maxiter without meeting the convergence criterion")
    return (sweep.maxiter, false)
end

end # module
