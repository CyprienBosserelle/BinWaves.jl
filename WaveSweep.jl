# WaveSweep.jl : implicit, four-sweep Gauss-Seidel solver for the wave action balance equation.
# Load AFTER WaveAction.jl, in the same module (normally Main):
#     include("WaveAction.jl"); include("WaveSweep.jl"); using .WaveAction, .WaveSweep
isdefined(Main, :WaveAction) || include(joinpath(@__DIR__, "WaveAction.jl"))

"""
    WaveSweep

SWAN-style implicit solver working on the `WaveModel` of `WaveAction.jl` (same grid, depth/current environment,
boundaries, source-term physics and outputs).

Scheme (per grid cell, per frequency):
* backward Euler in time (non-stationary mode) or no time term (stationary mode);
* backward-space upwind (BSBT-like) in x and y, with the donor chosen from the sign of the face velocity;
* θ propagation (refraction) fully implicit: a cyclic tridiagonal system over the directions of the sweep;
* σ propagation (current / depth-change shift) implicit in the cell's own bin, with the neighbouring frequency taken
  from the latest iterate (frequencies are visited in increasing order, so the lower neighbour is already updated);
* source terms: dissipation (whitecapping, breaking, friction) linearised and treated implicitly (Patankar), wind input
  explicit with a growth limiter.

Four-direction Gauss-Seidel: the domain is swept four times, once per quadrant of propagation directions
(+x+y, -x+y, -x-y, +x-y), visiting the cells so that the upwind neighbours are already updated (the information of each
direction bin travels through the whole domain in one sweep). Only the bins that propagate into the quadrant of the sweep
are updated in that sweep. Cells are processed along anti-diagonal wavefronts so they can be threaded while giving exactly
the same result as the sequential lexicographic order (results do not depend on the number of threads).

Unconditionally stable in time: `dt` may be many times the explicit CFL limit.
"""
module WaveSweep

using Base.Threads
using ..WaveAction
import ..WaveAction: NG, apply_bc!, breakingfraction

export SweepParams, four_sweeps!, step_implicit!, run_implicit!, solve_stationary!, hs_tm

"""
    SweepParams(; kwargs...)

* `w_theta`, `w_sigma`: weight of central differencing in θ / σ propagation (0 = pure upwind, 1 = central;
  SWAN uses 0.5 by default). Values > 0 are second-order but not strictly monotone (negative values are clipped).
* `iters`: number of four-sweep iterations per time step (non-stationary mode).
* `maxiter`, `drel`, `dabs`, `npnts`: stationary-mode convergence: relative/absolute change of Hs and Tm01 between
  iterations must be below `max(dabs, drel*value)` in a fraction `npnts` of the wet cells.
"""
Base.@kwdef struct SweepParams
    w_theta::Float64 = 0.5
    w_sigma::Float64 = 0.5
    iters::Int = 1
    maxiter::Int = 100
    drel::Float64 = 0.01
    dabs::Float64 = 0.005
    npnts::Float64 = 0.995
end

# ---------------------------------------------------------------------------------------------------
# Per-thread workspace
# ---------------------------------------------------------------------------------------------------
mutable struct Workspace
    a::Vector{Float64}; b::Vector{Float64}; c::Vector{Float64}; r::Vector{Float64}
    x::Vector{Float64}; z::Vector{Float64}; bb::Vector{Float64}; gam::Vector{Float64}; u::Vector{Float64}
    act::Vector{Bool}
end
Workspace(ns, nth) = Workspace([zeros(nth) for _ in 1:9]..., fill(false, nth))

# Thomas algorithm: a (sub, a[2:n]), b (diag), c (super, c[1:n-1])
function tridiag!(x, a, b, c, r, gam, n)
    bet = b[1]
    x[1] = r[1] / bet
    @inbounds for j in 2:n
        gam[j] = c[j-1] / bet
        bet = b[j] - a[j] * gam[j]
        x[j] = (r[j] - a[j] * x[j-1]) / bet
    end
    @inbounds for j in n-1:-1:1
        x[j] -= gam[j+1] * x[j+1]
    end
    return x
end

# Thomas algorithm on the segment lo:hi of the arrays (same conventions)
function tridiag_seg!(x, a, b, c, r, gam, lo, hi)
    bet = b[lo]
    x[lo] = r[lo] / bet
    @inbounds for q in lo+1:hi
        gam[q] = c[q-1] / bet
        bet = b[q] - a[q] * gam[q]
        x[q] = (r[q] - a[q] * x[q-1]) / bet
    end
    @inbounds for q in hi-1:-1:lo
        x[q] -= gam[q+1] * x[q+1]
    end
    return x
end

# Cyclic tridiagonal system (Sherman-Morrison). Row i: a[i] x[i-1] + b[i] x[i] + c[i] x[i+1] = r[i] with
# periodic indices: a[1] multiplies x[n] and c[n] multiplies x[1].
function cyclic!(ws::Workspace, n)
    a, b, c, r, x, z, bb, gam, u = ws.a, ws.b, ws.c, ws.r, ws.x, ws.z, ws.bb, ws.gam, ws.u
    alpha = c[n]; beta = a[1]
    gamma = -b[1]
    @inbounds for k in 1:n
        bb[k] = b[k]; u[k] = 0.0
    end
    bb[1] = b[1] - gamma
    bb[n] = b[n] - alpha * beta / gamma
    u[1] = gamma; u[n] = alpha
    tridiag!(x, a, bb, c, r, gam, n)
    tridiag!(z, a, bb, c, u, gam, n)
    fact = (x[1] + beta * x[n] / gamma) / (1.0 + z[1] + beta * z[n] / gamma)
    @inbounds for k in 1:n
        x[k] -= fact * z[k]
    end
    return x
end

# ---------------------------------------------------------------------------------------------------
# Dissipation rate of every cell (whitecapping + depth-induced breaking + bottom friction), from the current iterate.
# Computed once per four-sweep iteration (not once per sweep and cell visit) using the precomputed wave numbers;
# the wind input is evaluated in the sweep because it only needs local scalars.
# ---------------------------------------------------------------------------------------------------
function update_dissipation!(m::WaveModel)
    p, sp, e, gr, N = m.params, m.spec, m.env, m.grid, m.N
    (p.whitecap || p.breaking || p.friction) || return m
    ns, nth, g = sp.ns, sp.nth, p.g
    sig, dsig, dth = sp.sig, sp.dsig, sp.dth
    @threads :static for j in NG+1:NG+gr.ny
        @inbounds for i in NG+1:NG+gr.nx
            e.wet[i, j] || continue
            d = e.d[i, j]
            Etot = 0.0; m1 = 0.0; mm1 = 0.0; sk = 0.0
            for is in 1:ns
                σ = sig[is]
                s = 0.0
                for it in 1:nth
                    s += N[is, it, i, j]
                end
                Es = σ * s * dth * dsig[is]
                Etot += Es; m1 += σ * Es; mm1 += Es / σ; sk += sqrt(e.kk[is, i, j]) * Es
            end
            haveE = Etot > 1e-12
            sbar = haveE ? m1 / Etot : 1.0
            stil = haveE ? Etot / mm1 : 1.0
            ktil = haveE ? (sk / Etot)^2 : 1.0
            steep = ktil * sqrt(Etot)
            brate = 0.0
            if p.breaking && haveE
                Hm = p.gamma_bj * d
                brate = 0.25 * p.alpha_bj * breakingfraction(sqrt(8Etot) / Hm) * (sbar / 2π) * Hm^2 / Etot
            end
            wcfac = (p.whitecap && haveE) ? p.Cds * (steep / sqrt(p.stpm))^p.p_ds * stil : 0.0
            for is in 1:ns
                σ = sig[is]; k = e.kk[is, i, j]
                dec = brate
                p.whitecap && haveE && (dec += wcfac * ((1 - p.delta_ds) + p.delta_ds * k / ktil) * (k / ktil))
                p.friction && (dec += p.Cbf * σ^2 / g^2 * e.sh2i[is, i, j])
                e.dec[is, i, j] = dec
            end
        end
    end
    return m
end

# ---------------------------------------------------------------------------------------------------
# Steady garden-sprinkler-effect diffusion (stationary mode)
#   The along-axis diffusion (Dxx, Dyy) of ∇·(D∇N) enters the cell equation implicitly (diagonal + known neighbours);
#   the cross terms (Dxy) are explicit, evaluated from the iterate at the start of each sweep into m.dN.
#   Only used with a time-step-independent filter time scale (GSEParams cfl > 0 or tau > 0).
# ---------------------------------------------------------------------------------------------------
@inline function gse_DssDnn(g::GSEParams, cg, dxm, kdc, dth)
    τ = g.tau > 0 ? g.tau : g.cfl * dxm / cg
    return g.alpha_s^2 / 3 * (kdc * cg)^2 * τ, g.alpha_n^2 / 3 * (cg * dth)^2 * τ
end

function gse_cross!(dN, N, m::WaveModel, g::GSEParams)
    sp, e, gr = m.spec, m.env, m.grid
    ns, nth = sp.ns, sp.nth
    nx, ny = gr.nx, gr.ny
    idx, idy = 1 / gr.dx, 1 / gr.dy
    dxm = min(gr.dx, gr.dy)
    kdc = 0.5 * (sp.fac - 1 / sp.fac)
    dth, cth, sth, wet = sp.dth, sp.cth, sp.sth, e.wet
    @inline gyc(is, it, i, j) = (N[is, it, i, j+1] - N[is, it, i, j-1]) * (0.5 * idy)
    @inline gxc(is, it, i, j) = (N[is, it, i+1, j] - N[is, it, i-1, j]) * (0.5 * idx)
    @threads :static for j in NG+1:NG+ny
        @inbounds for i in NG+1:NG+nx
            if !wet[i, j]
                for it in 1:nth, is in 1:ns
                    dN[is, it, i, j] = 0.0
                end
                continue
            end
            wxl = wet[i-1, j]; wxr = wet[i+1, j]; wyb = wet[i, j-1]; wyt = wet[i, j+1]
            for it in 1:nth
                cs = cth[it] * sth[it]
                for is in 1:ns
                    Fl = 0.0; Fr = 0.0; Fb = 0.0; Ft = 0.0
                    if wxl
                        Dss, Dnn = gse_DssDnn(g, e.cgx[is, i, j], dxm, kdc, dth)
                        Fl = -(Dss - Dnn) * cs * 0.5 * (gyc(is, it, i - 1, j) + gyc(is, it, i, j))
                    end
                    if wxr
                        Dss, Dnn = gse_DssDnn(g, e.cgx[is, i+1, j], dxm, kdc, dth)
                        Fr = -(Dss - Dnn) * cs * 0.5 * (gyc(is, it, i, j) + gyc(is, it, i + 1, j))
                    end
                    if wyb
                        Dss, Dnn = gse_DssDnn(g, e.cgy[is, i, j], dxm, kdc, dth)
                        Fb = -(Dss - Dnn) * cs * 0.5 * (gxc(is, it, i, j - 1) + gxc(is, it, i, j))
                    end
                    if wyt
                        Dss, Dnn = gse_DssDnn(g, e.cgy[is, i, j+1], dxm, kdc, dth)
                        Ft = -(Dss - Dnn) * cs * 0.5 * (gxc(is, it, i, j) + gxc(is, it, i, j + 1))
                    end
                    dN[is, it, i, j] = -((Fr - Fl) * idx + (Ft - Fb) * idy)
                end
            end
        end
    end
    return dN
end

# ---------------------------------------------------------------------------------------------------
# Implicit update of one cell for one sweep
# ---------------------------------------------------------------------------------------------------
@inline H(c) = c > 0 ? 1.0 : 0.0

function sweep_cell!(m::WaveModel, ws::Workspace, i, j, sx, sy, dtinv, Nn, sw::SweepParams, gsel)
    sp, p, e, gr, N = m.spec, m.params, m.env, m.grid, m.N
    ns, nth = sp.ns, sp.nth
    idx, idy = 1 / gr.dx, 1 / gr.dy
    cth, sth, dsig, dth = sp.cth, sp.sth, sp.dsig, sp.dth
    wθ, wσ = sw.w_theta, sw.w_sigma
    dorefr = e.refr[i, j]                      # refraction / shift coefficients are non-zero in this cell
    doshift = e.has_shift && dorefr
    dissip = p.whitecap || p.breaking || p.friction
    g = p.g
    Ufl, Ufr, Vfb, Vft = e.Ufx[i, j], e.Ufx[i+1, j], e.Vfy[i, j], e.Vfy[i, j+1]
    a, b, c, r, act, x = ws.a, ws.b, ws.c, ws.r, ws.act, ws.x
    # Zero-gradient (OpenBC) boundaries are treated implicitly: the ghost value equals the cell's own value, so the
    # flux through that face is c_face * N_cell for either sign of c_face. (A lagged ghost copy would make the
    # iteration converge extremely slowly.)  Imposed-spectrum / zero-inflow ghosts are known values.
    gl = i == NG + 1 && m.bc[1] isa OpenBC
    gr_ = i == NG + gr.nx && m.bc[2] isa OpenBC
    gb = j == NG + 1 && m.bc[3] isa OpenBC
    gt = j == NG + gr.ny && m.bc[4] isa OpenBC
    dogse = gsel !== nothing
    if dogse   # steady GSE diffusion: faces towards dry cells and zero-gradient (open) boundaries carry no flux
        dxm = min(gr.dx, gr.dy); kdc = 0.5 * (sp.fac - 1 / sp.fac)
        wxl = e.wet[i-1, j] && !gl; wxr = e.wet[i+1, j] && !gr_; wyb = e.wet[i, j-1] && !gb; wyt = e.wet[i, j+1] && !gt
    end
    # wind scalars of the cell
    u10x, u10y = e.u10x[i, j], e.u10y[i, j]
    U10 = hypot(u10x, u10y)
    dowind = p.wind && U10 > 0.1
    ustar = 0.0; cw = 1.0; sw_ = 0.0; σPM = 1.0
    if dowind
        Cd = U10 < 7.5 ? 1.2875e-3 : (0.8 + 0.065U10) * 1e-3
        ustar = U10 * sqrt(Cd)
        cw, sw_ = u10x / U10, u10y / U10
        σPM = 2π * 0.13 * g / (28ustar)
    end

    @inbounds for is in 1:ns
        σ = sp.sig[is]
        ssh = e.ssh[is, i, j]
        cgxl, cgxr, cgyb, cgyt = e.cgx[is, i, j], e.cgx[is, i+1, j], e.cgy[is, i, j], e.cgy[is, i, j+1]
        dσ = dsig[is]
        # ---- which directions propagate into the quadrant of this sweep
        nact = 0; t0 = 0
        for it in 1:nth
            c0, s0 = cth[it], sth[it]
            isact = (((cgxl + cgxr) * c0 + Ufl + Ufr) >= 0 ? 1 : -1) == sx && (((cgyb + cgyt) * s0 + Vfb + Vft) >= 0 ? 1 : -1) == sy
            act[it] = isact
            if isact; nact += 1; else t0 = it; end
        end
        nact == 0 && continue
        ring = nact == nth                       # every direction active: cyclic system (rare, strong currents)
        ring && (t0 = 0)
        decis = dissip ? e.dec[is, i, j] : 0.0
        Alin = 0.0; dNmax = Inf; cph = 1.0
        if dowind
            σ2 = σ * σ
            Alin = 1.5e-3 / (2π * g^2) * exp(-(σ / σPM)^-4)
            dNmax = p.Cl * p.alpha_pm * g^2 / (2π * σ2 * σ2 * σ2)
            cph = σ / e.kk[is, i, j]
        end
        # ---- coefficients, stored in position order q (it = t0 + q, wrapping): the inactive bin t0 sits at q = nth,
        #      so the active directions form runs that never wrap and are solved as plain tridiagonal systems
        for q in 1:nth
            it = t0 + q; it > nth && (it -= nth)
            a[q] = 0.0; c[q] = 0.0
            act[it] || continue
            c0, s0 = cth[it], sth[it]
            cxl = cgxl * c0 + Ufl; cxr = cgxr * c0 + Ufr
            cyb = cgyb * s0 + Vfb; cyt = cgyt * s0 + Vft
            Nc = N[is, it, i, j]
            diag = decis
            rhs = 0.0
            if dowind
                cosd = c0 * cw + s0 * sw_
                uc = ustar * max(0.0, cosd)
                rhs += Alin * uc * uc * uc * uc / σ + max(0.0, 0.25 * p.rhoratio * (28ustar / cph * cosd - 1)) * σ * Nc
            end
            if dtinv > 0
                diag += dtinv
                rhs += Nn[is, it, i, j] * dtinv
            end
            # ---- physical space: upwind donors chosen from face velocities
            if cxl > 0 && !gl; rhs += cxl * N[is, it, i-1, j] * idx; else diag -= cxl * idx; end
            if cxr > 0 || gr_; diag += cxr * idx; else rhs -= cxr * N[is, it, i+1, j] * idx; end
            if cyb > 0 && !gb; rhs += cyb * N[is, it, i, j-1] * idy; else diag -= cyb * idy; end
            if cyt > 0 || gt; diag += cyt * idy; else rhs -= cyt * N[is, it, i, j+1] * idy; end
            # ---- steady GSE diffusion: along-axis part implicit, cross part explicit (m.dN)
            if dogse
                if wxl
                    Dss, Dnn = gse_DssDnn(gsel, cgxl, dxm, kdc, dth); Dxx = (Dss * c0 * c0 + Dnn * s0 * s0) * idx * idx
                    diag += Dxx; rhs += Dxx * N[is, it, i-1, j]
                end
                if wxr
                    Dss, Dnn = gse_DssDnn(gsel, cgxr, dxm, kdc, dth); Dxx = (Dss * c0 * c0 + Dnn * s0 * s0) * idx * idx
                    diag += Dxx; rhs += Dxx * N[is, it, i+1, j]
                end
                if wyb
                    Dss, Dnn = gse_DssDnn(gsel, cgyb, dxm, kdc, dth); Dyy = (Dss * s0 * s0 + Dnn * c0 * c0) * idy * idy
                    diag += Dyy; rhs += Dyy * N[is, it, i, j-1]
                end
                if wyt
                    Dss, Dnn = gse_DssDnn(gsel, cgyt, dxm, kdc, dth); Dyy = (Dss * s0 * s0 + Dnn * c0 * c0) * idy * idy
                    diag += Dyy; rhs += Dyy * N[is, it, i, j+1]
                end
                rhs += m.dN[is, it, i, j]
            end
            # ---- theta (implicit, in the tridiagonal matrix); refraction coefficients precomputed per cell and direction
            if dorefr
                cthp = ssh * e.gdp[it, i, j] + e.gUp[it, i, j]
                cthm = ssh * e.gdm[it, i, j] + e.gUm[it, i, j]
                diag += (cthp * ((1 - wθ) * H(cthp) + 0.5wθ) - cthm * ((1 - wθ) * (1 - H(cthm)) + 0.5wθ)) / dth
                c[q] = cthp * ((1 - wθ) * (1 - H(cthp)) + 0.5wθ) / dth           # coefficient of N[it+1]
                a[q] = -cthm * ((1 - wθ) * H(cthm) + 0.5wθ) / dth                # coefficient of N[it-1]
            end
            # ---- sigma (own bin implicit, neighbours from latest iterate)
            if doshift
                csp = e.csig[1, is+1, i, j] + e.csig[2, is+1, i, j] * c0 * c0 + e.csig[3, is+1, i, j] * c0 * s0 + e.csig[4, is+1, i, j] * s0 * s0
                csm = e.csig[1, is, i, j] + e.csig[2, is, i, j] * c0 * c0 + e.csig[3, is, i, j] * c0 * s0 + e.csig[4, is, i, j] * s0 * s0
                if is == ns
                    diag += max(csp, 0.0) / dσ
                else
                    diag += csp * ((1 - wσ) * H(csp) + 0.5wσ) / dσ
                    rhs -= csp * ((1 - wσ) * (1 - H(csp)) + 0.5wσ) * N[is+1, it, i, j] / dσ
                end
                if is > 1
                    diag -= csm * ((1 - wσ) * (1 - H(csm)) + 0.5wσ) / dσ
                    rhs += csm * ((1 - wσ) * H(csm) + 0.5wσ) * N[is-1, it, i, j] / dσ
                end
            end
            b[q] = max(diag, 1e-12)
            r[q] = rhs
        end
        # ---- solve
        if ring
            cyclic!(ws, nth)
        else
            q = 1
            while q <= nth
                itq = t0 + q; itq > nth && (itq -= nth)
                if !act[itq]
                    q += 1; continue
                end
                lo = q
                while q + 1 <= nth
                    itn = t0 + q + 1; itn > nth && (itn -= nth)
                    act[itn] || break
                    q += 1
                end
                hi = q
                # neighbours outside the run are inactive: known values go to the right-hand side
                itlo = t0 + lo; itlo > nth && (itlo -= nth)
                ithi = t0 + hi; ithi > nth && (ithi -= nth)
                itprev = itlo == 1 ? nth : itlo - 1
                itnext = ithi == nth ? 1 : ithi + 1
                r[lo] -= a[lo] * N[is, itprev, i, j]; a[lo] = 0.0
                r[hi] -= c[hi] * N[is, itnext, i, j]; c[hi] = 0.0
                tridiag_seg!(x, a, b, c, r, ws.gam, lo, hi)
                q += 1
            end
        end
        for q in 1:nth
            it = t0 + q; it > nth && (it -= nth)
            act[it] || continue
            v = max(x[q], 0.0)
            if dowind
                v = min(v, N[is, it, i, j] + dNmax)
            end
            N[is, it, i, j] = v
        end
    end
    return nothing
end

# One sweep for the quadrant (sx, sy), cells visited along anti-diagonal wavefronts of the sweep's own
# upwind ordering. Each cell reads only its 4 face neighbours (previous / next wavefront), so the result is
# identical to a sequential lexicographic Gauss-Seidel sweep for any number of threads.
function sweep!(m::WaveModel, sx, sy, dtinv, Nn, sw::SweepParams, wss, gsel = nothing)
    nx, ny = m.grid.nx, m.grid.ny
    wet = m.env.wet
    for k in 2:nx+ny
        ulo = max(1, k - ny); uhi = min(nx, k - 1)
        @threads :static for u in ulo:uhi
            v = k - u
            i = (sx > 0 ? u : nx + 1 - u) + NG
            j = (sy > 0 ? v : ny + 1 - v) + NG
            wet[i, j] && sweep_cell!(m, wss[threadid()], i, j, sx, sy, dtinv, Nn, sw, gsel)
        end
    end
    return m
end

"""
    four_sweeps!(m, dtinv, Nn, sw, wss, tbc)

One iteration: four sweeps over the quadrants (+,+), (−,+), (−,−), (+,−). `dtinv = 1/dt` (0 for stationary),
`Nn` the action at the previous time level, `tbc` the time at which boundary spectra are evaluated.
"""
function four_sweeps!(m::WaveModel, dtinv, Nn, sw::SweepParams, wss, tbc; gsel = nothing)
    for (sx, sy) in ((1, 1), (-1, 1), (-1, -1), (1, -1))
        apply_bc!(m.N, m, tbc)
        update_dissipation!(m)                  # dissipation rates from the latest iterate, once per sweep
        gsel === nothing || gse_cross!(m.dN, m.N, m, gsel)     # explicit cross-diffusion terms, from the current iterate
        sweep!(m, sx, sy, dtinv, Nn, sw, wss, gsel)
    end
    apply_bc!(m.N, m, tbc)
    return m
end

make_workspaces(m::WaveModel) = [Workspace(m.spec.ns, m.spec.nth) for _ in 1:Threads.maxthreadid()]

"""
    step_implicit!(m, dt, sw=SweepParams(); wss=make_workspaces(m))

One backward-Euler time step of length `dt` (any size) using `sw.iters` four-sweep iterations.
Uses `m.N1` as storage for the previous time level.
"""
function step_implicit!(m::WaveModel, dt, sw::SweepParams = SweepParams(); wss = make_workspaces(m))
    copyto!(m.N1, m.N)
    for _ in 1:sw.iters
        four_sweeps!(m, 1 / dt, m.N1, sw, wss, m.t + dt)
    end
    m.t += dt
    WaveAction.gse_step!(m, dt)          # garden-sprinkler-effect alleviation (no-op unless set_gse! was called)
    return m
end

"""
    run_implicit!(m; tend, dt = 10*stable_dt, sweep = SweepParams(), output_dt = nothing, callback = nothing, verbose = true)

Non-stationary run to absolute time `tend` with time step `dt` (unconditionally stable; accuracy is first order in time,
so choose `dt` a few to tens of times the explicit CFL step). The step is shortened to hit `tend` and every `output_dt`.
Returns the number of steps.
"""
function run_implicit!(m::WaveModel; tend, dt = nothing, sweep::SweepParams = SweepParams(),
                       output_dt = nothing, callback = nothing, verbose = true)
    dtmax = dt === nothing ? 10 * WaveAction.stable_dt(m, 0.5) : dt
    isfinite(dtmax) || error("no wet cells: nothing to compute")
    wss = make_workspaces(m)
    verbose && println("WaveSweep (implicit, 4 sweeps): dt = $(round(dtmax, digits = 3)) s " *
                       "($(round(dtmax / WaveAction.stable_dt(m, 0.5), digits = 1)) × explicit CFL step), threads = $(nthreads()), " *
                       "grid $(m.grid.nx)×$(m.grid.ny), spectrum $(m.spec.ns)×$(m.spec.nth)")
    nextout = output_dt === nothing ? Inf : m.t + output_dt
    callback === nothing || callback(m, m.t)
    nsteps = 0
    tol = 1e-9 * max(1.0, abs(tend))
    while m.t < tend - tol
        h = min(dtmax, tend - m.t, nextout - m.t)
        step_implicit!(m, h, sweep; wss = wss)
        nsteps += 1
        if m.t >= nextout - tol
            callback === nothing || callback(m, m.t)
            nextout += output_dt
        end
        all(isfinite, @view m.N[:, :, NG+1:NG+m.grid.nx, NG+1:NG+m.grid.ny]) || error("non-finite wave action at t = $(m.t) s")
    end
    return nsteps
end

"""
    hs_tm(m) -> (Hs, Tm01)

Hs and mean period fields (nx, ny); used for the stationary convergence test.
"""
function hs_tm(m::WaveModel)
    sp = m.spec
    nx, ny = m.grid.nx, m.grid.ny
    Hs = zeros(nx, ny); Tm = zeros(nx, ny)
    @threads :static for j in 1:ny
        for i in 1:nx
            m0 = 0.0; m1 = 0.0
            for it in 1:sp.nth, is in 1:sp.ns
                E = sp.sig[is] * m.N[is, it, i+NG, j+NG] * sp.dth * sp.dsig[is]
                m0 += E; m1 += sp.sig[is] * E
            end
            Hs[i, j] = 4sqrt(m0)
            Tm[i, j] = m1 > 1e-20 ? 2π * m0 / m1 : 0.0
        end
    end
    return Hs, Tm
end

"""
    solve_stationary!(m; sweep = SweepParams(), verbose = true) -> (iterations, converged)

Stationary solution (∂/∂t = 0) by repeated four-sweep iterations starting from the current state. If garden sprinkler alleviation
is on (`set_gse!`) it is included as a steady diffusion term (Ts = Δx/cg, i.e. `cfl` mode; a model with `cfl = 0, tau = 0` falls back to
`cfl = 1`). Converged when
Hs and Tm01 change by less than `max(dabs, drel·value)` between iterations in at least `npnts` of the wet cells
(SWAN-like criterion, without the curvature test). The boundary spectra are evaluated at the current model time.
"""
function solve_stationary!(m::WaveModel; sweep::SweepParams = SweepParams(), verbose::Bool = true)
    wss = make_workspaces(m)
    nx, ny = m.grid.nx, m.grid.ny
    wet = @view m.env.wet[NG+1:NG+nx, NG+1:NG+ny]
    nwet = count(wet)
    # Garden sprinkler alleviation enters the steady equation as a diffusion term; it needs a filter time scale that
    # does not depend on a time step (cfl > 0 or tau > 0). With Ts = dt there is no dt: fall back to cfl = 1.
    gsel = m.gse
    if gsel !== nothing && gsel.tau <= 0 && gsel.cfl <= 0
        verbose && println("  note: stationary mode has no time step; GSE filter uses cfl = 1 (Ts = Δx/cg)")
        gsel = GSEParams(alpha_s = gsel.alpha_s, alpha_n = gsel.alpha_n, cfl = 1.0, tau = 0.0, maxsub = gsel.maxsub)
    end
    verbose && gsel !== nothing && println("  garden sprinkler alleviation active (steady diffusion, alpha_s = $(gsel.alpha_s), alpha_n = $(gsel.alpha_n))")
    nwet == 0 && return (0, true)
    Hso, Tmo = hs_tm(m)
    for iter in 1:sweep.maxiter
        four_sweeps!(m, 0.0, m.N1, sweep, wss, m.t; gsel = gsel)
        Hs, Tm = hs_tm(m)
        ok = 0
        for j in 1:ny, i in 1:nx
            wet[i, j] || continue
            okH = abs(Hs[i, j] - Hso[i, j]) <= max(sweep.dabs, sweep.drel * Hs[i, j])
            okT = abs(Tm[i, j] - Tmo[i, j]) <= max(sweep.dabs, sweep.drel * Tm[i, j])
            ok += (okH && okT)
        end
        frac = ok / nwet
        verbose && println("  iteration $iter: $(round(100frac, digits = 2)) % of wet cells converged")
        Hso, Tmo = Hs, Tm
        iter > 1 && frac >= sweep.npnts && return (iter, true)
    end
    verbose && println("  stationary solver reached maxiter without meeting the convergence criterion")
    return (sweep.maxiter, false)
end

end # module
