# Local wind-sea look-up table for BinWaves: wind growth inside the model domain (semi-enclosed water bodies).
# Included by src/BinWaves.jl, after kp.jl.
#
# The swell part of BinWaves (kp.jl) is a linear superposition of boundary cases. Wind growth is not linear in the wind
# speed (and depends on direction through the fetch geometry), so the local wind sea is handled separately:
#
#   1. training: one stationary run per (wind direction, wind speed) pair, uniform wind over the domain, no incoming waves
#      (ZeroBC on every side), whitecapping on. Each run stores the wind-sea m0 field and the full spectrum at the points.
#   2. prediction: for a wind time series (U10(t), dir(t)) the stored states are interpolated: circularly-linear in direction
#      between the two nearest trained directions, and in speed along a power-law (log m0 linear in log U, the default) or
#      a linear curve between the two nearest trained speeds, extrapolated with the end segment outside the trained range.
#   3. the wind sea is added to the swell reconstruction as variance (m0 and point spectra): Hs² = Hs²_swell + Hs²_wind.

"""
    WindKP

Wind-sea look-up table built by `build_wind_kp`: wind-sea variance (m²) for every trained (direction, speed) pair.
* `dirs`    trained wind directions (nautical, coming from, degrees, ascending in [0, 360))
* `speeds`  trained wind speeds U10 (m/s, ascending, > 0)
* `m0`      (ndir, nspeed, nx, ny) wind-sea m0 (m²; 0 on dry cells)
* `spec`    (ndir, nspeed, ns, nth, npoints) wind-sea variance per model spectral bin (m²) at `points`
* `done`    (ndir, nspeed) which runs have been computed
"""
struct WindKP
    x::Vector{Float64}
    y::Vector{Float64}
    depth::Matrix{Float64}
    sig::Vector{Float64}
    dsig::Vector{Float64}
    sigf::Vector{Float64}
    dth::Float64
    dirs::Vector{Float64}
    speeds::Vector{Float64}
    m0::Array{Float32,4}
    points::Vector{Tuple{Int,Int}}
    spec::Array{Float32,5}
    done::Matrix{Bool}
    info::Dict{String,Any}
end
Base.show(io::IO, k::WindKP) = print(io, "WindKP: ", count(k.done), "/", length(k.done), " runs computed (", length(k.dirs),
    " directions × ", length(k.speeds), " speeds ", round.(k.speeds; digits = 2), " m/s), grid ", length(k.x), "×", length(k.y),
    ", ", length(k.points), " spectral points")

_with(p::WaveParams; kw...) = WaveParams(; (f => (haskey(kw, f) ? kw[f] : getfield(p, f)) for f in fieldnames(WaveParams))...)

# ------------------------------------------------------------------------------------------------------------
# NetCDF storage (also used for checkpoints while build_wind_kp runs)
# ------------------------------------------------------------------------------------------------------------
function _create_wkp_nc(file::AbstractString, wk::WindKP; compress::Int = 1)
    isfile(file) && rm(file)
    nx, ny, ns, nth, nd, nu, np = length(wk.x), length(wk.y), length(wk.sig), round(Int, 2π / wk.dth), length(wk.dirs),
                                  length(wk.speeds), length(wk.points)
    fill0 = Dict{String,Any}("_FillValue" => 0f0)
    dirs_bin = [mod(270 - rad2deg((k - 1) * wk.dth), 360) for k in 1:nth]
    nccreate(file, "depth", "x", wk.x, Dict("units" => "m"), "y", wk.y, Dict("units" => "m"); atts = Dict("units" => "m"),
             t = NC_DOUBLE, compress = compress)
    nccreate(file, "sig", "freq", wk.sig ./ 2π, Dict("units" => "Hz"); atts = Dict("units" => "rad s-1"), t = NC_DOUBLE)
    nccreate(file, "dsig", "freq", ns; atts = Dict("units" => "rad s-1"), t = NC_DOUBLE)
    nccreate(file, "sigf", "freq_face", ns + 1; atts = Dict("units" => "rad s-1"), t = NC_DOUBLE)
    nccreate(file, "dir_from", "dir", dirs_bin, Dict("units" => "degree"); atts = Dict("units" => "degree"), t = NC_DOUBLE)
    nccreate(file, "dth", "scalar_dim", 1; atts = Dict("units" => "rad"), t = NC_DOUBLE)
    nccreate(file, "wind_dir", "wdir", wk.dirs, Dict("units" => "degree", "long_name" => "trained wind direction, nautical from");
             t = NC_DOUBLE)
    nccreate(file, "wind_speed", "wspd", wk.speeds, Dict("units" => "m s-1", "long_name" => "trained wind speed U10"); t = NC_DOUBLE)
    nccreate(file, "run_done", "wdir", nd, "wspd", nu; atts = Dict("long_name" => "1 when the run has been computed"), t = NC_INT)
    nccreate(file, "m0", "x", nx, "y", ny, "wdir", nd, "wspd", nu;
             atts = merge(fill0, Dict("long_name" => "wind-sea variance m0", "units" => "m2")), t = NC_FLOAT, compress = compress)
    if np > 0
        nccreate(file, "point_i", "point", np; t = NC_INT); nccreate(file, "point_j", "point", np; t = NC_INT)
        nccreate(file, "spec", "freq", ns, "dir", nth, "point", np, "wdir", nd, "wspd", nu;
                 atts = merge(fill0, Dict("long_name" => "wind-sea variance per spectral bin", "units" => "m2")),
                 t = NC_FLOAT, compress = compress)
    end
    ncwrite(wk.x, file, "x"); ncwrite(wk.y, file, "y"); ncwrite(wk.depth, file, "depth")
    ncwrite(wk.sig, file, "sig"); ncwrite(wk.dsig, file, "dsig"); ncwrite(wk.sigf, file, "sigf"); ncwrite([wk.dth], file, "dth")
    ncwrite(dirs_bin, file, "dir_from"); ncwrite(wk.dirs, file, "wind_dir"); ncwrite(wk.speeds, file, "wind_speed")
    if np > 0
        ncwrite(Int32.(first.(wk.points)), file, "point_i"); ncwrite(Int32.(last.(wk.points)), file, "point_j")
    end
    ga = Dict{String,Any}("title" => "BinWaves local wind-sea look-up table", "Conventions" => "CF-1.8",
        "source" => "BinWaves.jl / WaveAction.jl", "seconds" => Float64(get(wk.info, "seconds", 0.0)),
        "whitecap" => Int32(get(wk.info, "whitecap", true)), "breaking" => Int32(get(wk.info, "breaking", false)),
        "friction" => Int32(get(wk.info, "friction", false)), "history" => "created " * string(Dates_now()))
    ncputatt(file, "NC_GLOBAL", ga)
    return file
end

function _write_runs!(file::AbstractString, wk::WindKP, idx)
    nx, ny, ns, nth, np = length(wk.x), length(wk.y), length(wk.sig), round(Int, 2π / wk.dth), length(wk.points)
    for (a, b) in idx
        ncwrite(reshape(wk.m0[a, b, :, :], nx, ny, 1, 1), file, "m0"; start = [1, 1, a, b], count = [nx, ny, 1, 1])
        np > 0 && ncwrite(reshape(wk.spec[a, b, :, :, :], ns, nth, np, 1, 1), file, "spec";
                          start = [1, 1, 1, a, b], count = [ns, nth, np, 1, 1])
        ncwrite(reshape(Int32[wk.done[a, b]], 1, 1), file, "run_done"; start = [a, b], count = [1, 1])
    end
    return nothing
end

"""
    save_wind_kp_netcdf(file, wk; compress = 1)

Write a (complete or partial) `WindKP` to a NetCDF file (dimensions x, y, freq, dir, point, wdir, wspd).
"""
function save_wind_kp_netcdf(file::AbstractString, wk::WindKP; compress::Int = 1)
    _create_wkp_nc(file, wk; compress = compress)
    _write_runs!(file, wk, [Tuple(I) for I in findall(wk.done)])
    return file
end

"""
    load_wind_kp_netcdf(file) -> WindKP

Read a file written by `save_wind_kp_netcdf` or by `build_wind_kp(...; checkpoint = file)`.
"""
function load_wind_kp_netcdf(file::AbstractString)
    isfile(file) || error("no such file: $file")
    rd(v) = ncread(file, v)
    x, y, depth = Float64.(rd("x")), Float64.(rd("y")), Float64.(rd("depth"))
    sig, dsig, sigf, dth = Float64.(rd("sig")), Float64.(rd("dsig")), Float64.(rd("sigf")), Float64(rd("dth")[1])
    dirs, speeds = Float64.(rd("wind_dir")), Float64.(rd("wind_speed"))
    nd, nu = length(dirs), length(speeds)
    done = reshape(Int.(rd("run_done")), nd, nu) .== 1
    m0 = permutedims(Float32.(rd("m0")), (3, 4, 1, 2))
    haspts = NetCDF.open(file) do nc_; haskey(nc_.vars, "spec"); end
    if haspts
        pts = [(Int(a), Int(b)) for (a, b) in zip(rd("point_i"), rd("point_j"))]
        spec = permutedims(Float32.(rd("spec")), (4, 5, 1, 2, 3))
    else
        pts = Tuple{Int,Int}[]; spec = zeros(Float32, nd, nu, length(sig), round(Int, 2π / dth), 0)
    end
    ga = NetCDF.open(file) do nc_; Dict{String,Any}(nc_.gatts); end
    info = Dict{String,Any}("seconds" => Float64(get(ga, "seconds", 0.0)), "whitecap" => get(ga, "whitecap", 1) != 0,
                            "breaking" => get(ga, "breaking", 0) != 0, "friction" => get(ga, "friction", 0) != 0)
    wk = WindKP(x, y, depth, sig, dsig, sigf, dth, dirs, speeds, m0, pts, spec, done, info)
    all(done) || @warn "$file holds $(count(done))/$(length(done)) wind runs; finish it with build_wind_kp(...; checkpoint=\"$file\")"
    return wk
end

function _check_compatible(old::WindKP, new::WindKP, file)
    ok = size(old.m0) == size(new.m0) && size(old.spec) == size(new.spec) && old.dth ≈ new.dth && old.sig ≈ new.sig &&
         old.dirs ≈ new.dirs && old.speeds ≈ new.speeds && old.points == new.points && old.x ≈ new.x && old.y ≈ new.y &&
         old.depth ≈ new.depth
    ok || error("checkpoint $file was made with a different grid, depth, spectral grid, wind directions / speeds or points; " *
                "delete it or pass resume = false")
end

# ------------------------------------------------------------------------------------------------------------
# training: one stationary run per (direction, speed)
# ------------------------------------------------------------------------------------------------------------
"""
    build_wind_kp(m; dirs = 0:22.5:337.5, speeds = [5, 10, 15, 20, 25, 30], points = [], whitecap = true,
                  sweep = SweepParams(w_theta = 0, w_sigma = 0, dabs = 1e-4, maxiter = 200), verbose = true,
                  checkpoint = nothing, checkpoint_every = 10, resume = true) -> WindKP

Train the local wind-sea table: for every wind direction (nautical, coming from, degrees) and wind speed U10 (m/s), run the
stationary solver on `m` with a uniform wind, no incoming waves (`ZeroBC()` on every side) and wind input on; whitecapping
is switched on as well unless `whitecap = false` (without a dissipation the stationary wind sea has no balance). The other
physics (breaking, friction, refraction, currents, GSE) are used as set on the model. The model's own boundary conditions,
wind field and parameters are restored afterwards.

* `speeds`  cover the range you will predict for: outside it the prediction extrapolates the end segments of the
            curve. Wind growth is far from linear in U, so use at least 4-6 speeds (more where the sea state changes regime,
            e.g. where it becomes depth limited).
* `dirs`    the prediction interpolates linearly between neighbouring directions; 16 (22.5°) is a good default, use more for
            narrow inlets or strongly direction-dependent fetch.
* `sweep`   stationary solver settings. The default absolute tolerance is tight (0.1 mm): with the solver's usual 5 mm a light
            wind's slowly growing sea "converges" early (measured: half the converged Hs at 5 m/s).
* `checkpoint`  NetCDF file updated every `checkpoint_every` runs; an existing file is resumed (`resume = true`).

The same `points` as the swell `build_kp` must be used to add the point spectra in `reconstruct`.
"""
function build_wind_kp(m::WaveModel; dirs = 0:22.5:337.5, speeds = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
                       points = Tuple{Int,Int}[], whitecap::Bool = true,
                       sweep = SweepParams(w_theta = 0.0, w_sigma = 0.0, dabs = 1e-4, maxiter = 200), verbose::Bool = true,
                       checkpoint = nothing, checkpoint_every::Int = 10, resume::Bool = true)
    sp, g = m.spec, m.grid
    ns, nth, nx, ny = sp.ns, sp.nth, g.nx, g.ny
    dirv = sort(unique(mod.(collect(Float64, dirs), 360.0)))
    spdv = sort(unique(collect(Float64, speeds)))
    isempty(dirv) && error("give at least one wind direction")
    isempty(spdv) && error("give at least one wind speed")
    first(spdv) > 0.1 || error("wind speeds must be > 0.1 m/s (the model switches wind input off below that)")
    whitecap || m.params.whitecap ||
        @warn "build_wind_kp with whitecapping off: the stationary wind sea is only bounded by breaking / friction / fetch"
    for (i, j) in points
        (1 <= i <= nx && 1 <= j <= ny) || error("point ($i,$j) outside the grid")
    end
    nd, nu, npt = length(dirv), length(spdv), length(points)
    D = m.env.d[NGH+1:NGH+nx, NGH+1:NGH+ny]
    p_run = _with(m.params; wind = true, whitecap = whitecap || m.params.whitecap)
    info = Dict{String,Any}("seconds" => 0.0, "whitecap" => p_run.whitecap, "breaking" => p_run.breaking,
                            "friction" => p_run.friction)
    wk = WindKP(copy(g.x), copy(g.y), copy(D), copy(sp.sig), copy(sp.dsig), copy(sp.sigf), sp.dth, dirv, spdv,
                zeros(Float32, nd, nu, nx, ny), collect(points), zeros(Float32, nd, nu, ns, nth, npt), fill(false, nd, nu), info)
    t_prev = 0.0
    if checkpoint !== nothing
        if resume && isfile(checkpoint)
            old = load_wind_kp_netcdf(checkpoint)
            _check_compatible(old, wk, checkpoint)
            wk.m0 .= old.m0; wk.spec .= old.spec; wk.done .= old.done
            t_prev = Float64(get(old.info, "seconds", 0.0))
            verbose && @printf("  resuming from %s: %d/%d wind runs already done\n", checkpoint, count(wk.done), nd * nu)
        else
            _create_wkp_nc(checkpoint, wk)
        end
    end
    orig_p, orig_bc = m.params, m.bc
    orig_u, orig_v = copy(m.env.u10x), copy(m.env.u10y)
    pending = Tuple{Int,Int}[]
    t0 = time()
    flush_ck!() = if checkpoint !== nothing && !isempty(pending)
        _write_runs!(checkpoint, wk, pending)
        ncputatt(checkpoint, "NC_GLOBAL", Dict{String,Any}("seconds" => t_prev + (time() - t0)))
        empty!(pending)
    end
    todo = [(a, b) for b in 1:nu for a in 1:nd if !wk.done[a, b]]
    try
        m.params = p_run
        set_boundaries!(m; left = ZeroBC(), right = ZeroBC(), bottom = ZeroBC(), top = ZeroBC())
        for (n, (a, b)) in enumerate(todo)
            initialize!(m; Hs = 0.0)
            set_wind!(m; U10 = spdv[b], dir = dirv[a])
            it, conv = solve_stationary!(m; sweep = sweep, verbose = false)
            conv || @warn @sprintf("wind run (%.1f°, %.1f m/s) did not converge in %d iterations", dirv[a], spdv[b], it)
            for j in 1:ny, i in 1:nx
                m.env.wet[NGH+i, NGH+j] || continue
                s = 0.0
                for it_ in 1:nth, is in 1:ns
                    s += sp.sig[is] * m.N[is, it_, NGH+i, NGH+j] * sp.dsig[is]
                end
                wk.m0[a, b, i, j] = s * sp.dth
            end
            for (q, (i, j)) in enumerate(points), it_ in 1:nth, is in 1:ns
                wk.spec[a, b, is, it_, q] = sp.sig[is] * m.N[is, it_, NGH+i, NGH+j] * sp.dsig[is] * sp.dth
            end
            wk.done[a, b] = true
            push!(pending, (a, b))
            length(pending) >= checkpoint_every && flush_ck!()
            if verbose && (n == 1 || n % max(1, length(todo) ÷ 20) == 0 || n == length(todo))
                el = time() - t0
                @printf("  wind run %d/%d (%.1f°, %.1f m/s, %d iterations)   %.1f s elapsed, ~%.0f s remaining\n", n, length(todo),
                        dirv[a], spdv[b], it, el, el / n * (length(todo) - n))
            end
        end
    finally
        flush_ck!()
        m.params = orig_p
        set_boundaries!(m; left = orig_bc[1], right = orig_bc[2], bottom = orig_bc[3], top = orig_bc[4])
        m.env.u10x .= orig_u; m.env.u10y .= orig_v
    end
    info["seconds"] = t_prev + (time() - t0)
    checkpoint === nothing || ncputatt(checkpoint, "NC_GLOBAL", Dict{String,Any}("seconds" => info["seconds"]))
    return wk
end

# ------------------------------------------------------------------------------------------------------------
# prediction: interpolate in direction and speed
# ------------------------------------------------------------------------------------------------------------
"""Two bracketing trained directions and the weight of the second one (circular, linear in angle)."""
function _dir_bracket(dirs::Vector{Float64}, d::Real)
    n = length(dirs)
    n == 1 && return (1, 1, 0.0)
    d = mod(Float64(d), 360.0)
    k = searchsortedlast(dirs, d)                 # dirs[k] <= d < dirs[k+1]  (k = 0 or n: wrap)
    a = k == 0 ? n : k
    b = a == n ? 1 : a + 1
    span = mod(dirs[b] - dirs[a], 360.0); span == 0 && (span = 360.0)
    return (a, b, mod(d - dirs[a], 360.0) / span)
end

"""Speed bracket (k, k+1) and position: log-U fraction for :loglog, U fraction for :linear (outside [0,1] = extrapolation)."""
function _speed_bracket(speeds::Vector{Float64}, U::Real, interp::Symbol)
    n = length(speeds)
    n == 1 && return (1, 1, 0.0)
    k = clamp(searchsortedlast(speeds, U), 1, n - 1)
    u1, u2 = speeds[k], speeds[k+1]
    s = interp === :loglog ? (log(U) - log(u1)) / (log(u2) - log(u1)) : (U - u1) / (u2 - u1)
    return (k, k + 1, s)
end

"""
Wind-sea variance at speed U from the trained values v1 (at speeds[k]) and v2 (at speeds[k+1]).
:loglog : m0 ∝ U^p along the segment (p from the two values, clamped to [0, pmax] for extrapolation);
:linear : straight line, and below the first trained speed a line through the origin.
`single` (one trained speed): v1 (U / U1)^2.
"""
function _speed_value(v1, v2, s, U, u1, u2, interp::Symbol; pmax = 4.0)
    if interp === :loglog
        (v1 > 0 && v2 > 0) || return max(zero(v1), v1 + s * (v2 - v1))   # zero (dry / sheltered): fall back to linear
        if 0 <= s <= 1
            return exp((1 - s) * log(v1) + s * log(v2))
        end
        lu = log(u2 / u1)                                  # s is in units of log(u2/u1): log m0 = log v1 + s log(v2/v1)
        L = clamp(log(v2 / v1), 0.0, pmax * lu)            # i.e. m0 ∝ U^p with p = log(v2/v1)/lu clamped to [0, pmax]
        return s < 0 ? v1 * exp(s * L) : v2 * exp((s - 1) * L)
    else
        if s < 0 && U < u1                           # below the trained range: towards zero at U = 0
            return v1 * max(U, 0.0) / u1
        end
        return max(zero(v1), v1 + s * (v2 - v1))
    end
end

function _wind_weights(wk::WindKP, U::Real, d::Real, interp::Symbol)
    a1, a2, wd = _dir_bracket(wk.dirs, d)
    k1, k2, s = _speed_bracket(wk.speeds, U, interp)
    return a1, a2, wd, k1, k2, s
end

"""
    wind_sea(wk, U10, dir; fields = true, interp = :loglog, umin = 1.0) -> (; m0, Hs, point_var)

Wind-sea prediction for a time series of uniform local wind: `U10[t]` (m/s) and `dir[t]` (nautical, coming from, degrees).
* `m0`, `Hs`   (nx, ny, nt) wind-sea variance (m²) and significant wave height (only if `fields = true`)
* `point_var`  (nt, ns, nth, npoints) wind-sea variance per model spectral bin at the stored points
* `interp`     `:loglog` (default; m0 follows a power law of U between trained speeds — fetch-limited growth is close to
               Hs ∝ U, fully developed to Hs ∝ U²) or `:linear`. Outside the trained speeds the end segment is extrapolated
               (log-log slope clamped to m0 ∝ U⁴ at most); for `:linear` below the lowest speed a line to zero at U = 0.
* `umin`       below this speed the wind sea is set to zero.
Point spectra are interpolated as a weighted sum of the bracketing trained spectra, rescaled so that their variance follows
the same curve as m0 (the spectral shape is a blend of the two neighbouring states; its peak does not shift continuously).
"""
function wind_sea(wk::WindKP, U10::AbstractVector, dir::AbstractVector; fields::Bool = true, interp::Symbol = :loglog,
                  umin::Real = 1.0)
    interp in (:loglog, :linear) || error("interp must be :loglog or :linear")
    nt = length(U10); length(dir) == nt || error("U10 and dir must have the same length")
    all(wk.done) || @warn "only $(count(wk.done)) of $(length(wk.done)) wind runs are computed: the wind sea is incomplete"
    nx, ny = length(wk.x), length(wk.y)
    ns, nth, np = size(wk.spec, 3), size(wk.spec, 4), size(wk.spec, 5)
    m0 = fields ? zeros(Float32, nx, ny, nt) : zeros(Float32, 0, 0, 0)
    pv = zeros(nt, ns, nth, np)
    single = length(wk.speeds) == 1
    for t in 1:nt
        U = Float64(U10[t])
        (isfinite(U) && U >= umin && isfinite(dir[t])) || continue
        a1, a2, wd, k1, k2, s = _wind_weights(wk, U, dir[t], interp)
        u1, u2 = wk.speeds[k1], wk.speeds[k2]
        val(v1, v2) = single ? v1 * (U / u1)^2 : _speed_value(v1, v2, s, U, u1, u2, interp)
        if fields
            @inbounds for j in 1:ny, i in 1:nx
                va = val(Float64(wk.m0[a1, k1, i, j]), Float64(wk.m0[a1, k2, i, j]))
                vb = val(Float64(wk.m0[a2, k1, i, j]), Float64(wk.m0[a2, k2, i, j]))
                m0[i, j, t] = (1 - wd) * va + wd * vb
            end
        end
        sl = clamp(s, 0.0, 1.0)
        for q in 1:np, (a, w) in ((a1, 1 - wd), (a2, wd))
            w == 0 && continue
            S1 = @view wk.spec[a, k1, :, :, q]; S2 = @view wk.spec[a, k2, :, :, q]
            v1, v2 = Float64(sum(S1)), Float64(sum(S2))
            target = val(v1, v2)
            blend = (1 - sl) * v1 + sl * v2
            blend > 0 || continue
            f = w * target / blend
            @inbounds for it in 1:nth, is in 1:ns
                pv[t, is, it, q] += f * ((1 - sl) * S1[is, it] + sl * S2[is, it])
            end
        end
    end
    Hs = fields ? 4 .* sqrt.(max.(m0, 0f0)) : zeros(Float32, 0, 0, 0)
    return (; m0, Hs, point_var = pv)
end
