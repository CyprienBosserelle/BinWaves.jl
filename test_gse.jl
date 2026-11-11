# Tests of the garden-sprinkler-effect alleviation (anisotropic diffusion).   julia -t 4 test_gse.jl
include(joinpath(@__DIR__, "WaveAction.jl"))
include(joinpath(@__DIR__, "WaveSweep.jl"))
using .WaveAction, .WaveSweep
using Test, Printf

# moments of one spectral bin field over the interior
function moments(m, is, it)
    nx, ny = m.grid.nx, m.grid.ny
    S = Sx = Sy = 0.0
    for j in 1:ny, i in 1:nx
        n = m.N[is, it, i+2, j+2]; S += n; Sx += n * m.grid.x[i]; Sy += n * m.grid.y[j]
    end
    cx, cy = Sx / S, Sy / S
    xx = xy = yy = 0.0
    for j in 1:ny, i in 1:nx
        n = m.N[is, it, i+2, j+2]; X = m.grid.x[i] - cx; Y = m.grid.y[j] - cy
        xx += n * X * X; xy += n * X * Y; yy += n * Y * Y
    end
    return S, xx / S, xy / S, yy / S
end

function blob_model(; nth = 12, gse = nothing)
    nx = ny = 80; dx = 1000.0
    x = (0.5:1:nx) .* dx; y = (0.5:1:ny) .* dx
    spec = SpectralGrid(ns = 3, nth = nth, fmin = 0.1, fmax = 0.121)
    m = WaveModel(CartesianGrid(x, y); depth = fill(4000.0, nx, ny), spec = spec, params = WaveParams(refraction = false))
    xc, yc, σ = 40e3, 40e3, 5e3
    for j in 1:ny, i in 1:nx
        m.N[2, 2, i+2, j+2] = exp(-((x[i] - xc)^2 + (y[j] - yc)^2) / (2σ^2))      # bin (is=2, θ = 30°)
    end
    gse === nothing || set_gse!(m; gse...)
    return m
end

@testset "GSE alleviation" begin

@testset "diffusion coefficients: covariance growth = 2 D t (oblique bin, cross term included)" begin
    αs, αn, τ, dt = 3.0, 3.0, 100.0, 1000.0
    m = blob_model(gse = (alpha_s = αs, alpha_n = αn, cfl = 0.0, tau = τ))
    sp = m.spec
    θ = sp.dth                                    # bin 2 is at θ = Δθ = 30°
    c, s = cos(θ), sin(θ)
    _, cg, _ = wavekin(sp.sig[2], 4000.0)
    kd = 0.5 * (sp.fac - 1 / sp.fac)
    Dss = αs^2 * (kd * cg)^2 * τ / 3
    Dnn = αn^2 * (cg * sp.dth)^2 * τ / 3
    Dxx, Dyy, Dxy = Dss * c^2 + Dnn * s^2, Dss * s^2 + Dnn * c^2, (Dss - Dnn) * c * s
    S0, xx0, xy0, yy0 = moments(m, 2, 2)
    nsub = gse_step!(m, dt)
    S1, xx1, xy1, yy1 = moments(m, 2, 2)
    @printf("   %d sub-steps; growth/(2 t): xx %.1f (D %.1f)  yy %.1f (D %.1f)  xy %.1f (D %.1f)\n", nsub,
            (xx1 - xx0) / 2dt, Dxx, (yy1 - yy0) / 2dt, Dyy, (xy1 - xy0) / 2dt, Dxy)
    @test isapprox((xx1 - xx0) / (2dt), Dxx; rtol = 0.02)
    @test isapprox((yy1 - yy0) / (2dt), Dyy; rtol = 0.02)
    @test isapprox((xy1 - xy0) / (2dt), Dxy; rtol = 0.02)
    @test isapprox(S1, S0; rtol = 1e-7)           # conservative (ADI path: clipping of tiny negatives only)
end

@testset "off by default, and switching off restores the original behaviour" begin
    m = blob_model()
    @test m.gse === nothing && gse_step!(m, 100.0) == 0
    S0 = moments(m, 2, 2)
    set_gse!(m; alpha_s = 3.0, alpha_n = 3.0); set_gse!(m, nothing)
    gse_step!(m, 100.0)
    @test moments(m, 2, 2) == S0
end

@testset "per-frequency time scale (cfl mode) makes the smoothing independent of the solver step" begin
    function run_split(nsteps; kw...)
        m = blob_model(gse = (alpha_s = 3.0, alpha_n = 3.0, cfl = 1.0, kw...))
        for _ in 1:nsteps; gse_step!(m, 2000.0 / nsteps); end
        return m
    end
    # explicit sub-stepping path (forced): tight agreement
    m1, m4 = run_split(1; adi_above = typemax(Int) ÷ 2, maxsub = 100000), run_split(4; adi_above = typemax(Int) ÷ 2, maxsub = 100000)
    a = m1.N[2, 2, 3:82, 3:82]; b = m4.N[2, 2, 3:82, 3:82]
    rel = maximum(abs.(a .- b)) / maximum(a)
    @printf("   explicit: 1 step of 2000 s vs 4 steps of 500 s: max difference %.2e of the peak\n", rel)
    @test rel < 0.01
    # ADI path (default for large dt): backward-Euler/splitting error, looser but bounded
    A1, A4 = run_split(1), run_split(4)
    ra = maximum(abs.(A1.N[2, 2, 3:82, 3:82] .- a)) / maximum(a)
    @printf("   ADI (1 step) vs explicit reference: %.2e of the peak\n", ra)
    @test ra < 0.08
    @test minimum(A1.N) >= 0
    # ADI thread determinism is covered in the transport testset; with Ts = dt the result depends on the step (documented)
    mdt = blob_model(gse = (alpha_s = 3.0, alpha_n = 3.0, cfl = 0.0)); gse_step!(mdt, 2000.0)
    @test maximum(abs.(mdt.N[2, 2, 3:82, 3:82] .- a)) / maximum(a) > 1e-3
end

@testset "coupled with transport: conservation and positivity (explicit and implicit solvers)" begin
    nx, ny = 60, 50; dx = 100e3
    x = (0.5:1:nx) .* dx; y = (0.5:1:ny) .* dx
    spec = SpectralGrid(ns = 8, nth = 24, fmin = 0.06, fmax = 0.14)
    for solver in (:explicit, :implicit)
        m = WaveModel(CartesianGrid(x, y); depth = fill(4000.0, nx, ny), spec = spec, params = WaveParams(refraction = false), Hs0 = 0.0)
        for j in 22:29, i in 25:34, it in 1:24, is in 1:8
            m.N[is, it, i+2, j+2] = 0.5 * exp(-((it - 2)^2) / 4)
        end
        set_gse!(m; alpha_s = 3.0, alpha_n = 3.0)
        a0 = total_action(m)
        solver == :explicit ? run!(m; tend = 86400.0, verbose = false) : run_implicit!(m; tend = 86400.0, dt = 2000.0, verbose = false)
        a1 = total_action(m)
        interior = @view m.N[:, :, 3:nx+2, 3:ny+2]
        @printf("   %-9s: relative change of total action %.2e, min N %.1e\n", solver, (a1 - a0) / a0, minimum(interior))
        # the packet stays away from the open boundaries; the loss is the (tiny) clipping of undershoots and top-of-spectrum outflow
        @test abs(a1 - a0) / a0 < 1e-3
        @test minimum(interior) >= 0
    end
end

@testset "stationary solver includes the GSE filter (slit-through-barrier case)" begin
    dx = 10e3; nx, ny = 70, 50
    x = (0.5:1:nx) .* dx; y = (0.5:1:ny) .* dx
    ib0, ib1, gap = 10, 11, 24:27
    depth = fill(4000.0, nx, ny)
    for j in 1:ny, i in ib0:ib1; j in gap || (depth[i, j] = 0.0); end
    xf, xb = (ib0 - 1) * dx, ib1 * dx; ylo, yhi = (first(gap) - 1) * dx, last(gap) * dx
    spec = SpectralGrid(ns = 3, nth = 36, fmin = 0.10, fmax = 0.121); sD = 3.0; Hs_in = 2.0
    # exact (geometric optics): fraction of the incident directional distribution whose ray passes through the gap
    nθ = 7200; th = range(-π, π; length = nθ + 1); th = 0.5 .* (th[1:end-1] .+ th[2:end]); dθ = 2π / nθ
    D = @. max(cos(th / 2), 0.0)^(2sD); D ./= sum(D) * dθ
    Hex = zeros(nx, ny)
    for j in 1:ny, i in 1:nx
        x[i] <= xb && continue
        f = 0.0
        for (k, θ) in enumerate(th)
            cos(θ) > 1e-9 || continue
            yf = y[j] - (x[i] - xf) * tan(θ); yb = y[j] - (x[i] - xb) * tan(θ)
            (ylo <= yf <= yhi && ylo <= yb <= yhi) && (f += D[k] * dθ)
        end
        Hex[i, j] = Hs_in * sqrt(f)
    end
    reg = [(i, j) for j in 1:ny, i in 1:nx if x[i] > xb + 5dx && Hex[i, j] > 0.05]
    rms(H) = sqrt(sum((H[i, j] - Hex[i, j])^2 for (i, j) in reg) / length(reg))
    function solve(gse)
        m = WaveModel(CartesianGrid(x, y); depth = depth, spec = spec, params = WaveParams(refraction = false, hmin = 0.5))
        set_boundaries!(m; left = SpectrumBC(Hs = Hs_in, Tp = 9.0, dir = 270.0, spread = sD), right = ZeroBC(), bottom = ZeroBC(), top = ZeroBC())
        gse === nothing || set_gse!(m; gse...)
        it, conv = solve_stationary!(m; sweep = SweepParams(maxiter = 200), verbose = false)
        return m, it, conv
    end
    m0, it0, c0 = solve(nothing)
    m3, it3, c3 = solve((alpha_s = 3.0, alpha_n = 3.0))
    e0, e3 = rms(bulk(m0).Hs), rms(bulk(m3).Hs)
    @printf("   stationary: rms error vs exact %.3f m (%d its) without GSE, %.3f m (%d its) with α = 3\n", e0, it0, e3, it3)
    @test c0 && c3
    @test e3 < 0.6 * e0                                   # the filter removes most of the sprinkler error
    @test minimum(@view m3.N[:, :, 3:nx+2, 3:ny+2]) >= 0
    # the steady-diffusion form must agree with a long non-stationary run that uses the split-step filter
    mt = WaveModel(CartesianGrid(x, y); depth = depth, spec = spec, params = WaveParams(refraction = false, hmin = 0.5))
    set_boundaries!(mt; left = SpectrumBC(Hs = Hs_in, Tp = 9.0, dir = 270.0, spread = sD), right = ZeroBC(), bottom = ZeroBC(), top = ZeroBC())
    set_gse!(mt; alpha_s = 3.0, alpha_n = 3.0)
    run_implicit!(mt; tend = 4.0e5, dt = 4000.0, verbose = false)
    a, b = bulk(m3).Hs, bulk(mt).Hs
    dr = sqrt(sum((a[i, j] - b[i, j])^2 for j in 1:ny, i in 1:nx if x[i] > xb + 5dx) / count(i -> x[i] > xb + 5dx, 1:nx) / ny)
    @printf("   stationary vs long time-dependent run: rms difference %.4f m\n", dr)
    @test dr < 0.01
end

end
