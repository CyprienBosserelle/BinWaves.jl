# Stationary garden sprinkler test: a broad-direction swell meets a dry barrier with a narrow gap, which acts as a point
# source. In deep water the stationary solution is purely geometric: the energy at (x,y) is the fraction of the incident
# directional distribution whose ray (travelling back along -θ) passes through the gap. The discrete direction bins give
# separated rays (the garden sprinkler effect). Compared with the exact fraction.
#   julia -t auto gse_stationary_test.jl
include(joinpath(@__DIR__, "WaveAction.jl"))
include(joinpath(@__DIR__, "WaveSweep.jl"))
using .WaveAction, .WaveSweep
using Printf

dx = 10e3
nx, ny = 70, 50
x = (0.5:1:nx) .* dx; y = (0.5:1:ny) .* dx
ib0, ib1 = 10, 11                                   # barrier cells (x index), 2 cells thick
gap = 24:27                                         # gap rows (4 cells = 40 km)
depth = fill(4000.0, nx, ny)
for j in 1:ny, i in ib0:ib1
    j in gap || (depth[i, j] = 0.0)
end
xf, xb = (ib0 - 1) * dx, ib1 * dx                   # x of the front / back faces of the barrier
ylo, yhi = (first(gap) - 1) * dx, last(gap) * dx

spec = SpectralGrid(ns = 3, nth = 36, fmin = 0.10, fmax = 0.121)   # 10° bins
sD = 3.0                                            # directional spreading exponent of the incident swell
Hs_in = 2.0

function exact_Hs()
    # continuous incident distribution D(θ) ∝ cos^{2s}(θ/2), normalised over the circle; waves travel towards +x (270° nautical)
    nθ = 7200; th = range(-π, π; length = nθ + 1); th = 0.5 .* (th[1:end-1] .+ th[2:end]); dθ = 2π / nθ
    D = @. max(cos(th / 2), 0.0)^(2sD); D ./= sum(D) * dθ
    Hs = zeros(nx, ny)
    for j in 1:ny, i in 1:nx
        x[i] <= xb && continue
        f = 0.0
        for (k, θ) in enumerate(th)
            c = cos(θ); c > 1e-9 || continue
            t = tan(θ)
            yf = y[j] - (x[i] - xf) * t; yb = y[j] - (x[i] - xb) * t   # ray position at the front and back faces of the barrier
            (ylo <= yf <= yhi && ylo <= yb <= yhi) && (f += D[k] * dθ)
        end
        Hs[i, j] = Hs_in * sqrt(f)
    end
    return Hs
end

function solve(gse)
    m = WaveModel(CartesianGrid(x, y); depth = depth, spec = spec, params = WaveParams(refraction = false, hmin = 0.5))
    set_boundaries!(m; left = SpectrumBC(Hs = Hs_in, Tp = 9.0, dir = 270.0, spread = sD), right = ZeroBC(), bottom = ZeroBC(), top = ZeroBC())
    gse === nothing || set_gse!(m; gse...)
    iters, conv = solve_stationary!(m; sweep = SweepParams(maxiter = 200), verbose = false)
    return m, iters, conv
end

if abspath(PROGRAM_FILE) == @__FILE__
    Hex = exact_Hs()
    region = [(i, j) for j in 1:ny, i in 1:nx if x[i] > xb + 5dx && Hex[i, j] > 0.05]
    err(Hs) = (sqrt(sum((Hs[i, j] - Hex[i, j])^2 for (i, j) in region) / length(region)), maximum(abs(Hs[i, j] - Hex[i, j]) for (i, j) in region))
    results = Dict{String,Matrix{Float64}}("exact" => Hex)
    for (label, gse) in (("stationary, no GSE alleviation", nothing), ("stationary, GSE α = 1", (alpha_s = 1.0, alpha_n = 1.0)),
                         ("stationary, GSE α = 2", (alpha_s = 2.0, alpha_n = 2.0)), ("stationary, GSE α = 3", (alpha_s = 3.0, alpha_n = 3.0)))
        m, iters, conv = solve(gse)
        Hs = bulk(m).Hs
        r, mx = err(Hs)
        @printf("%-32s %3d iterations (converged = %s)   rms error %.3f m   max error %.3f m   max Hs %.3f m\n", label, iters, conv, r, mx, maximum(Hs[ib1+6:end, :])); flush(stdout)
        results[label] = Hs
    end
    open(joinpath(@__DIR__, "gse_stationary_fields.csv"), "w") do io
        names = collect(keys(results)); println(io, "x_km,y_km,", join(replace.(names, "," => ";"), ","))
        for j in 1:ny, i in 1:nx
            println(io, x[i] / 1e3, ",", y[j] / 1e3, ",", join([results[n][i, j] for n in names], ","))
        end
    end
end
