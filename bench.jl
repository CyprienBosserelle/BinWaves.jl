# Micro-benchmarks of the hot paths.   julia -t 1 bench.jl [path_to_dir_with_WaveAction.jl_and_WaveSweep.jl]
dir = length(ARGS) > 0 ? ARGS[1] : @__DIR__
include(joinpath(dir, "WaveAction.jl")); include(joinpath(dir, "WaveSweep.jl"))
using .WaveAction, .WaveSweep, Printf

function make(; nx = 100, ny = 60, ns = 16, nth = 24, flat = false, sources = false, currents = false)
    dx = 25.0
    x = (0.5:1:nx) .* dx; y = (0.5:1:ny) .* dx
    depth = [flat ? 50.0 : max(14.0 - 13.0 * xi / (nx * dx), 0.0) - 3exp(-((xi - 0.5nx * dx)^2 + (yi - 0.5ny * dx)^2) / (300^2)) for xi in x, yi in y]
    depth = max.(depth, 0.0)
    U = currents ? [0.3 * sin(2π * yi / (ny * dx)) for xi in x, yi in y] : nothing
    spec = SpectralGrid(ns = ns, nth = nth, fmin = 0.06, fmax = 0.4)
    par = sources ? WaveParams(breaking = true, friction = true, whitecap = true, wind = true, U10 = 12.0, Wdir = 260.0, hmin = 0.3) : WaveParams(hmin = 0.3)
    m = WaveModel(CartesianGrid(x, y); depth = depth, U = U, spec = spec, params = par, Hs0 = 1.0, Tp0 = 9.0, dir0 = 260.0)
    set_boundaries!(m; left = SpectrumBC(Hs = 1.5, Tp = 9.0, dir = 260.0))
    return m
end

tm(f, n) = (f(); minimum(@elapsed(f()) for _ in 1:n))
function report(label, m)
    ncell = m.grid.nx * m.grid.ny; nb = m.spec.ns * m.spec.nth
    dt = stable_dt(m, 0.4)
    wss = WaveSweep.make_workspaces(m)
    t_tend = tm(() -> WaveAction.tendency!(m.dN, m.N, m), 3)
    t_step = tm(() -> step!(m, dt), 3)
    t_four = tm(() -> WaveSweep.four_sweeps!(m, 1 / (5dt), m.N1, SweepParams(), wss, m.t), 3)
    t_src = tm(() -> WaveAction.sources!(m, dt), 3)
    @printf("%-44s tendency %7.1f ms | explicit step %7.1f ms | 4 sweeps %7.1f ms | sources %7.1f ms   [%d cells × %d bins]\n",
            label, 1e3t_tend, 1e3t_step, 1e3t_four, 1e3t_src, ncell, nb)
end
if abspath(PROGRAM_FILE) == @__FILE__
    println("threads = ", Threads.nthreads())
    report("variable depth, no currents, no sources", make())
    report("flat deep water, no currents, no sources", make(flat = true))
    report("variable depth + currents", make(currents = true))
    report("variable depth + all sources", make(sources = true))
end
