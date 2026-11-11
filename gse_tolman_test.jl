# Garden sprinkler effect test case of Booij & Holthuijsen (1987) / Tolman (2002) / Popinet et al. (2010):
# a Gaussian blob of swell (Hs0 = 2.5 m, sigma_r = 150 km), cos^2 directional distribution about 30°, Gaussian frequency
# distribution (0.1 ± 0.02 Hz), propagating 5 days in deep water without sources. Directional resolution 15°, γ = 1.1,
# dx = 100 km. The continuous solution is known in closed form (each (f,θ) component is just translated by cg(f) t).
#   julia -t auto gse_tolman_test.jl            (writes gse_fields.csv for plotting)
include(joinpath(@__DIR__, "WaveAction.jl"))
include(joinpath(@__DIR__, "WaveSweep.jl"))
using .WaveAction, .WaveSweep
using Printf, DelimitedFiles

const g = 9.81
const H0, σr, fl, fr, θl = 2.5, 150e3, 0.1, 0.02, deg2rad(30.0)
A(x, y) = H0^2 / 16 * exp(-(x^2 + y^2) / σr^2)

# ---- mesh: 48 x 38 cells of 100 km, containing the original domain ----------------------------------------------------
dx = 100e3
x = collect(-550e3:dx:4150e3); y = collect(-550e3:dx:3150e3)
nx, ny = length(x), length(y)
depth = fill(4000.0, nx, ny)

# ---- spectral grid: f1 = 0.04 Hz, ratio γ = 1.1, Δθ = 15°  (nth overridable) --------------------------------------------
function mkspec(nth; γ = 1.1, f1 = 0.04, nf = 18)
    SpectralGrid(ns = nf, nth = nth, fmin = f1, fmax = f1 * γ^(nf - 1))
end

# bin-averaged initial spectrum  N = A(x) Sbin Dbin / (σ dσ Δθ)  (energy in a bin = σ N dσ dθ)
# integral of the (unit-area) Gaussian frequency distribution over [a, b] by midpoint quadrature
function gauss_bin(a, b; n = 400)
    h = (b - a) / n
    return sum(exp(-((a + (k - 0.5) * h) - fl)^2 / (2fr^2)) for k in 1:n) * h / (fr * sqrt(2π))
end
Dcdf(θ) = (θ <= -π/2 ? 0.0 : θ >= π/2 ? 1.0 : (θ / 2 + sin(2θ) / 4) * (2 / π) + 0.5)   # CDF of (2/π) cos²θ on [-π/2, π/2]
function initial!(m, spec)
    rf = spec.rf
    Sbin = [gauss_bin(spec.sig[is] / rf / 2π, spec.sig[is] * rf / 2π) for is in 1:spec.ns]
    Dbin = map(1:spec.nth) do it
        θ0 = (it - 1) * spec.dth - θl
        θ0 = atan(sin(θ0), cos(θ0))
        lo, hi = θ0 - spec.dth / 2, θ0 + spec.dth / 2
        Dcdf(hi) - Dcdf(lo)                                            # bins never straddle ±π/2 problems for 15° / 6° bins
    end
    for j in 1:ny, i in 1:nx, it in 1:spec.nth, is in 1:spec.ns
        m.N[is, it, i+2, j+2] = A(x[i], y[j]) * Sbin[is] * Dbin[it] / (spec.sig[is] * spec.dsig[is] * spec.dth)
    end
    return m
end

# ---- exact (continuous) solution at time t: m0(x) = ∫∫ A(x - cg(f) t (cosθ, sinθ)) S(f) D(θ) df dθ ------------------
function exact_Hs(spec, t; nf = 160, nθ = 320)
    flo, fhi = spec.sig[1] / spec.rf / 2π, spec.sig[end] * spec.rf / 2π
    fs = range(flo, fhi; length = nf + 1); fs = 0.5 .* (fs[1:end-1] .+ fs[2:end]); dfq = (fhi - flo) / nf
    Sf = @. exp(-(fs - fl)^2 / (2fr^2)) / (fr * sqrt(2π))
    ths = range(θl - π/2, θl + π/2; length = nθ + 1); ths = 0.5 .* (ths[1:end-1] .+ ths[2:end]); dθq = π / nθ
    Dth = @. (2 / π) * cos(ths - θl)^2
    cgs = @. g / (4π * fs)
    Hs = zeros(nx, ny)
    Threads.@threads :static for j in 1:ny
        for i in 1:nx
            s = 0.0
            for (a, f) in enumerate(fs), (b, θ) in enumerate(ths)
                d = cgs[a] * t
                s += A(x[i] - d * cos(θ), y[j] - d * sin(θ)) * Sf[a] * Dth[b] * dfq * dθq
            end
            Hs[i, j] = 4sqrt(s)
        end
    end
    return Hs
end

function errstats(Hs, Hex)
    w = Hex .> 0.05
    rms = sqrt(sum((Hs[w] .- Hex[w]) .^ 2) / count(w))
    return rms, maximum(abs.(Hs[w] .- Hex[w])), maximum(Hs)
end

function run_case(label, spec; solver = :explicit, gse = nothing, tend = 5 * 86400.0, dtfac = 1)
    m = WaveModel(CartesianGrid(x, y); depth = depth, spec = spec, params = WaveParams(order = 2, refraction = false))
    initial!(m, spec)
    gse === nothing || set_gse!(m; gse...)
    t0 = time()
    if solver == :explicit
        run!(m; tend = tend, cfl = 0.4, verbose = false)
    else
        dt = dtfac * stable_dt(m, 0.5)
        run_implicit!(m; tend = tend, dt = dt, verbose = false)
    end
    return m, time() - t0
end

if abspath(PROGRAM_FILE) == @__FILE__
    spec15 = mkspec(24)
    tend = 5 * 86400.0
    cache = joinpath(@__DIR__, "gse_exact.csv")
    if isfile(cache)
        Hex = readdlm(cache, ',')
    else
        Hex = exact_Hs(spec15, tend)
        writedlm(cache, Hex, ',')
    end
    @printf("exact solution: max Hs = %.3f m\n\n", maximum(Hex)); flush(stdout)
    results = Dict{String,Matrix{Float64}}("exact" => Hex)
    cases = [
        ("explicit, no GSE alleviation   ", mkspec(24), :explicit, nothing),
        ("explicit, GSE α = 3            ", mkspec(24), :explicit, (alpha_s = 3.0, alpha_n = 3.0)),
        ("explicit, GSE α = 4            ", mkspec(24), :explicit, (alpha_s = 4.0, alpha_n = 4.0)),
        ("explicit, GSE α = 3, Ts = dt   ", mkspec(24), :explicit, (alpha_s = 3.0, alpha_n = 3.0, cfl = 0.0)),
        ("implicit 4-sweep, no GSE       ", mkspec(24), :implicit, nothing),
        ("implicit 4-sweep, GSE α = 3    ", mkspec(24), :implicit, (alpha_s = 3.0, alpha_n = 3.0)),
        ("explicit, no GSE, Δθ = 3°      ", mkspec(120), :explicit, nothing),
    ]
    for (label, spec, solver, gse) in cases
        m, wall = run_case(label, spec; solver = solver, gse = gse)
        Hs = bulk(m).Hs
        rms, mx, hmax = errstats(Hs, Hex)
        @printf("%s  rms error %.3f m   max error %.3f m   max Hs %.3f m   (%.0f s)\n", label, rms, mx, hmax, wall)
        results[strip(label)] = Hs; flush(stdout)
    end
    open(joinpath(@__DIR__, "gse_fields.csv"), "w") do io
        names = collect(keys(results)); println(io, "x_km,y_km,", join(replace.(names, "," => ";"), ","))
        for j in 1:ny, i in 1:nx
            println(io, x[i] / 1e3, ",", y[j] / 1e3, ",", join([results[n][i, j] for n in names], ","))
        end
    end
end
