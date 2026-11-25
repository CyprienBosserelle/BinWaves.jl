# BinWaves transfer-coefficient (KP) build / reconstruction. Included by src/BinWaves.jl.

const NGH = WaveAction.NG

# ------------------------------------------------------------------------------------------------------------
# Model interface: a regular-grid `WaveModel` or a block-mesh `BlockModel`.
#   regular grid: fields are (nx, ny), points are grid indices (i, j) or (x, y) coordinates (nearest cell)
#   block mesh:   fields are (ncells, 1) in `cell_index` order, points are (x, y) coordinates, stored as (block, cell in block)
# ------------------------------------------------------------------------------------------------------------
_fieldsize(m::WaveModel) = (m.grid.nx, m.grid.ny)
_fieldsize(m::BlockModel) = (ncells(m), 1)
_coords(m::WaveModel) = (copy(m.grid.x), copy(m.grid.y))
_coords(m::BlockModel) = cell_centres(m.mesh)
_depthfield(m::WaveModel) = m.env.d[NGH+1:NGH+m.grid.nx, NGH+1:NGH+m.grid.ny]
_depthfield(m::BlockModel) = reshape(cell_depth(m), :, 1)
_meshinfo!(info, m::WaveModel) = info
_meshinfo!(info, m::BlockModel) = (info["mesh"] = deepcopy(m.mesh); info)

function _resolve_points(m::WaveModel, points)
    out = Tuple{Int,Int}[]
    nx, ny = m.grid.nx, m.grid.ny
    for p in points
        if p[1] isa Integer && p[2] isa Integer
            i, j = Int(p[1]), Int(p[2])
        else
            i = argmin(abs.(m.grid.x .- p[1])); j = argmin(abs.(m.grid.y .- p[2]))
        end
        (1 <= i <= nx && 1 <= j <= ny) || error("point ($i,$j) outside the grid")
        push!(out, (i, j))
    end
    return out
end
function _resolve_points(m::BlockModel, points)
    out = Tuple{Int,Int}[]
    for p in points
        (p[1] isa Integer && p[2] isa Integer) && error("on a block mesh give the points as (x, y) coordinates (Float64), got $p")
        r = locate(m.mesh, p[1], p[2])
        r === nothing && error("point $p is outside the mesh")
        k, i, j = r
        push!(out, (k, (j - 1) * m.mesh.bs + i))
    end
    return out
end

"""
Variance m0 and the energy-weighted direction vector (Σ E cos θ, Σ E sin θ; θ mathematical, propagating-to) of every cell,
times `scale`, into `m0`, `mc`, `ms` (fieldsize; `mc = ms = nothing` skips the direction moments). Dry cells get 0.
"""
function _moments!(m0::AbstractMatrix, mc, ms, m::WaveModel, scale)
    sp = m.spec
    for j in 1:m.grid.ny, i in 1:m.grid.nx
        _cellmom!(m0, mc, ms, (i, j), m, sp, NGH + i, NGH + j, scale)
    end
    return m0
end
function _moments!(m0::AbstractMatrix, mc, ms, m::BlockModel, scale)
    sp, bs = m.spec, m.mesh.bs
    for (k, b) in enumerate(m.blocks), j in 1:bs, i in 1:bs
        _cellmom!(m0, mc, ms, (cell_index(m.mesh, k, i, j), 1), b, sp, NGH + i, NGH + j, scale)
    end
    return m0
end
@inline function _cellmom!(m0, mc, ms, I, b::WaveModel, sp, ig, jg, scale)
    if !b.env.wet[ig, jg]
        m0[I...] = 0; mc === nothing || (mc[I...] = 0; ms[I...] = 0)
        return
    end
    s0 = 0.0; sc = 0.0; ss = 0.0
    @inbounds for it in 1:sp.nth
        e = 0.0
        for is in 1:sp.ns
            e += sp.sig[is] * b.N[is, it, ig, jg] * sp.dsig[is]
        end
        s0 += e; sc += e * sp.cth[it]; ss += e * sp.sth[it]
    end
    f = sp.dth * scale
    m0[I...] = s0 * f
    mc === nothing || (mc[I...] = sc * f; ms[I...] = ss * f)
    return
end

"""
Mean direction (nautical, coming from, degrees) and directional spread (degrees, σ = √(2(1 − r)), r = |(a, b)| / m0: the
circular-moment definition of Kuik et al. 1988, as in SWAN's DSPR) from m0 and the direction vector (a, b). NaN without energy.
"""
@inline function _dir_spread(m0, a, b)
    m0 > 1e-12 || return (NaN32, NaN32)
    r = min(sqrt(a^2 + b^2) / m0, 1.0)
    return (Float32(mod(270 - rad2deg(atan(b, a)), 360)), Float32(rad2deg(sqrt(2 * (1 - r)))))
end

"""Variance per spectral bin (ns, nth) at a stored point, times `scale`."""
_binvar!(out, m::WaveModel, p, scale) = _binvar!(out, m, m.spec, p[1] + NGH, p[2] + NGH, scale)
function _binvar!(out, m::BlockModel, p, scale)
    bs = m.mesh.bs
    k, c = p
    return _binvar!(out, m.blocks[k], m.spec, (c - 1) % bs + 1 + NGH, (c - 1) ÷ bs + 1 + NGH, scale)
end
function _binvar!(out, b::WaveModel, sp, ig, jg, scale)
    for it in 1:sp.nth, is in 1:sp.ns
        out[is, it] = sp.sig[is] * b.N[is, it, ig, jg] * sp.dsig[is] * sp.dth * scale
    end
    return out
end

# ---- mesh in NetCDF files ----
_hasmesh(info) = haskey(info, "mesh")
# dimension names / coordinate values of the field dimensions
_xydims(info, x, y) = _hasmesh(info) ? ("cell", collect(1.0:length(x)), "one", [1.0]) : ("x", x, "y", y)

function _nc_write_mesh!(file, mesh::BlockMesh, x, y)
    nccreate(file, "cell_x", "cell", length(x); atts = Dict("units" => "m", "long_name" => "cell centre x"), t = NC_DOUBLE)
    nccreate(file, "cell_y", "cell", length(y); atts = Dict("units" => "m", "long_name" => "cell centre y"), t = NC_DOUBLE)
    nb = nblocks(mesh)
    for v in ("mesh_level", "mesh_bx", "mesh_by")
        nccreate(file, v, "block", nb; t = NC_INT)
    end
    ncwrite(Float64.(x), file, "cell_x"); ncwrite(Float64.(y), file, "cell_y")
    ncwrite(Int32.(mesh.level), file, "mesh_level"); ncwrite(Int32.(mesh.bx), file, "mesh_bx"); ncwrite(Int32.(mesh.by), file, "mesh_by")
    ncputatt(file, "NC_GLOBAL", Dict{String,Any}("mesh_x0" => mesh.x0, "mesh_y0" => mesh.y0, "mesh_dx0" => mesh.dx0,
        "mesh_bs" => Int32(mesh.bs), "mesh_nbx" => Int32(mesh.nbx), "mesh_nby" => Int32(mesh.nby), "mesh_maxlevel" => Int32(mesh.maxlevel),
        "mesh_note" => "block-adaptive mesh: fields are on dimension 'cell' (block by block, i fastest); see BinWaves.WaveBlocks"))
end
function _nc_read_mesh(file)
    has = NetCDF.open(file) do nc_; haskey(nc_.vars, "mesh_level"); end
    has || return nothing
    ga = NetCDF.open(file) do nc_; Dict{String,Any}(nc_.gatts); end
    rd(v) = Int.(ncread(file, v))
    mesh = BlockMesh(ga["mesh_x0"], ga["mesh_y0"], ga["mesh_dx0"], ga["mesh_bs"], ga["mesh_nbx"], ga["mesh_nby"], ga["mesh_maxlevel"],
                     rd("mesh_level"), rd("mesh_bx"), rd("mesh_by"))
    return mesh, Float64.(ncread(file, "cell_x")), Float64.(ncread(file, "cell_y"))
end

struct BinWavesKP
    x::Vector{Float64}
    y::Vector{Float64}
    depth::Matrix{Float64}
    sig::Vector{Float64}
    dsig::Vector{Float64}
    sigf::Vector{Float64}
    dth::Float64
    fgroups::Vector{UnitRange{Int}}
    dgroups::Vector{Vector{Int}}                 # model direction-bin indices of each direction group (consecutive)
    cases::Vector{Tuple{Int,Int}}                # (frequency group, direction group) of each case
    Vin::Vector{Float64}                         # input variance of each case (m²)
    kp_m0::Array{Float32,3}                      # (ncase, nx, ny)  output m0 / input variance
    kp_cos::Array{Float32,3}                     # (ncase, nx, ny)  output Σ E cos θ / input variance  (θ math, going to)
    kp_sin::Array{Float32,3}                     # (ncase, nx, ny)  output Σ E sin θ / input variance;  both (ncase, 0, 0) if not stored
    points::Vector{Tuple{Int,Int}}
    kp_spec::Array{Float32,4}                    # (ncase, ns, nth, npoints) output variance per bin / input variance
    done::Vector{Bool}                           # which cases have been computed
    info::Dict{String,Any}
end
Base.show(io::IO, k::BinWavesKP) = print(io, "BinWavesKP: ", count(k.done), "/", length(k.cases), " cases computed (", length(k.fgroups), " freq groups × ",
    length(k.dgroups), " dir groups), ", _hasmesh(k.info) ? "block mesh of $(size(k.kp_m0, 2)) cells" : "grid $(length(k.x))×$(length(k.y))",
    ", ", length(k.points), " spectral points", _hasdir(k) ? ", direction moments" : "")

"""True when the coefficients include the direction moments (gridded mean direction and spread in `reconstruct`)."""
_hasdir(k::BinWavesKP) = size(k.kp_cos, 2) > 0

save_kp(file, kp::BinWavesKP) = serialize(file, kp)
load_kp(file) = deserialize(file)


# ------------------------------------------------------------------------------------------------------------
# NetCDF storage (also used for checkpoints while build_kp runs)
# ------------------------------------------------------------------------------------------------------------
_nth(kp::BinWavesKP) = round(Int, 2π / kp.dth)

"""Create an empty KP file with the right shape (unwritten cases read back as 0)."""
function _create_kp_nc(file::AbstractString, kp::BinWavesKP; compress::Int = 1)
    isfile(file) && rm(file)
    nx, ny, ns, nth, nc, np = size(kp.kp_m0, 2), size(kp.kp_m0, 3), length(kp.sig), _nth(kp), length(kp.cases), length(kp.points)
    dX, xv, dY, yv = _xydims(kp.info, kp.x, kp.y)
    fill0 = Dict{String,Any}("_FillValue" => 0f0)
    xa = Dict("units" => "m"); fa = Dict("units" => "Hz", "long_name" => "model frequency bin centres")
    da = Dict("units" => "degree", "long_name" => "model direction bins, nautical coming-from")
    dirs = [mod(270 - rad2deg((k - 1) * kp.dth), 360) for k in 1:nth]
    nccreate(file, "depth", dX, xv, xa, dY, yv, Dict("units" => "m"); atts = Dict("units" => "m"), t = NC_DOUBLE, compress = compress)
    nccreate(file, "sig", "freq", kp.sig ./ 2π, fa; atts = Dict("units" => "rad s-1", "long_name" => "angular frequency of the bin"), t = NC_DOUBLE)
    nccreate(file, "dsig", "freq", ns; atts = Dict("units" => "rad s-1", "long_name" => "bin width"), t = NC_DOUBLE)
    nccreate(file, "sigf", "freq_face", ns + 1; atts = Dict("units" => "rad s-1", "long_name" => "bin faces"), t = NC_DOUBLE)
    nccreate(file, "dir_from", "dir", dirs, da; atts = Dict("units" => "degree"), t = NC_DOUBLE)
    nccreate(file, "dth", "scalar_dim", 1; atts = Dict("units" => "rad", "long_name" => "direction bin width"), t = NC_DOUBLE)
    ng, nh = length(kp.fgroups), length(kp.dgroups)
    nccreate(file, "fgroup_first", "fgroup", ng; t = NC_INT); nccreate(file, "fgroup_last", "fgroup", ng; t = NC_INT)
    nccreate(file, "dgroup_first", "dgroup", nh; t = NC_INT); nccreate(file, "dgroup_len", "dgroup", nh; t = NC_INT)
    nccreate(file, "case_fgroup", "case", nc; t = NC_INT); nccreate(file, "case_dgroup", "case", nc; t = NC_INT)
    nccreate(file, "case_done", "case", nc; atts = Dict("long_name" => "1 when the case has been computed"), t = NC_INT)
    nccreate(file, "Vin", "case", nc; atts = Dict("units" => "m2", "long_name" => "input variance of the case"), t = NC_DOUBLE)
    nccreate(file, "kp_m0", dX, nx, dY, ny, "case", nc;
             atts = merge(fill0, Dict("long_name" => "output m0 / input variance", "units" => "1")), t = NC_FLOAT, compress = compress)
    if _hasdir(kp)
        for (v, ln) in (("kp_cos", "output sum of E cos(theta) / input variance (theta: mathematical, propagating-to)"),
                        ("kp_sin", "output sum of E sin(theta) / input variance (theta: mathematical, propagating-to)"))
            nccreate(file, v, dX, nx, dY, ny, "case", nc; atts = merge(fill0, Dict("long_name" => ln, "units" => "1")),
                     t = NC_FLOAT, compress = compress)
        end
    end
    if np > 0
        nccreate(file, "point_i", "point", np; t = NC_INT); nccreate(file, "point_j", "point", np; t = NC_INT)
        nccreate(file, "kp_spec", "freq", ns, "dir", nth, "point", np, "case", nc;
                 atts = merge(fill0, Dict("long_name" => "output variance per spectral bin / input variance", "units" => "1")),
                 t = NC_FLOAT, compress = compress)
    end
    # static content
    ncwrite(xv, file, dX); ncwrite(yv, file, dY); ncwrite(kp.depth, file, "depth")
    _hasmesh(kp.info) && _nc_write_mesh!(file, kp.info["mesh"], kp.x, kp.y)
    ncwrite(kp.sig, file, "sig"); ncwrite(kp.dsig, file, "dsig"); ncwrite(kp.sigf, file, "sigf"); ncwrite([kp.dth], file, "dth")
    ncwrite(dirs, file, "dir_from")
    ncwrite(Int32.(first.(kp.fgroups)), file, "fgroup_first"); ncwrite(Int32.(last.(kp.fgroups)), file, "fgroup_last")
    ncwrite(Int32.(first.(kp.dgroups)), file, "dgroup_first"); ncwrite(Int32.(length.(kp.dgroups)), file, "dgroup_len")
    ncwrite(Int32.(first.(kp.cases)), file, "case_fgroup"); ncwrite(Int32.(last.(kp.cases)), file, "case_dgroup")
    if np > 0
        ncwrite(Int32.(first.(kp.points)), file, "point_i"); ncwrite(Int32.(last.(kp.points)), file, "point_j")
    end
    ga = Dict{String,Any}("title" => "BinWaves transfer coefficients (KP)", "Conventions" => "CF-1.8",
        "source" => "BinWaves.jl / WaveAction.jl", "sides" => join(string.(get(kp.info, "sides", String[])), ","),
        "fgroup" => Int32(get(kp.info, "fgroup", 1)), "dgroup" => Int32(get(kp.info, "dgroup", 1)),
        "seconds" => Float64(get(kp.info, "seconds", 0.0)),
        "refraction" => Int32(get(kp.info, "refraction", true)), "breaking" => Int32(get(kp.info, "breaking", false)),
        "friction" => Int32(get(kp.info, "friction", false)), "wind" => Int32(get(kp.info, "wind", false)),
        "history" => "created " * string(Dates_now()))
    ncputatt(file, "NC_GLOBAL", ga)
    return file
end
Dates_now() = Base.Libc.strftime("%Y-%m-%d %H:%M:%S", time())

"""Write the listed cases (data, Vin, done flag) into an existing KP file."""
function _write_cases!(file::AbstractString, kp::BinWavesKP, idx)
    nx, ny, ns, nth, np = size(kp.kp_m0, 2), size(kp.kp_m0, 3), length(kp.sig), _nth(kp), length(kp.points)
    for c in idx
        ncwrite(reshape(Float32.(kp.kp_m0[c, :, :]), nx, ny, 1), file, "kp_m0"; start = [1, 1, c], count = [nx, ny, 1])
        if _hasdir(kp)
            ncwrite(reshape(Float32.(kp.kp_cos[c, :, :]), nx, ny, 1), file, "kp_cos"; start = [1, 1, c], count = [nx, ny, 1])
            ncwrite(reshape(Float32.(kp.kp_sin[c, :, :]), nx, ny, 1), file, "kp_sin"; start = [1, 1, c], count = [nx, ny, 1])
        end
        np > 0 && ncwrite(reshape(Float32.(kp.kp_spec[c, :, :, :]), ns, nth, np, 1), file, "kp_spec";
                          start = [1, 1, 1, c], count = [ns, nth, np, 1])
        ncwrite([kp.Vin[c]], file, "Vin"; start = [c], count = [1])
        ncwrite(Int32[kp.done[c]], file, "case_done"; start = [c], count = [1])
    end
    return nothing
end

"""
    save_kp_netcdf(file, kp; compress = 1)

Write a (complete or partial) `BinWavesKP` to a NetCDF file. Dimensions: x, y, freq, dir, point, case. File size is
`4·ncase·(k·nx·ny + ns·nth·npoints)` bytes before compression (k = 3 with direction moments, 1 without).
"""
function save_kp_netcdf(file::AbstractString, kp::BinWavesKP; compress::Int = 1)
    _create_kp_nc(file, kp; compress = compress)
    _write_cases!(file, kp, findall(kp.done))
    return file
end

"""
    load_kp_netcdf(file) -> BinWavesKP

Read a file written by `save_kp_netcdf` or by `build_kp(...; checkpoint = file)` and use it directly with `reconstruct`.
Cases that were not finished are flagged in `kp.done` (and `reconstruct` warns about them).
"""
function load_kp_netcdf(file::AbstractString)
    isfile(file) || error("no such file: $file")
    rd(v) = ncread(file, v)
    mr = _nc_read_mesh(file)
    x, y = mr === nothing ? (Float64.(rd("x")), Float64.(rd("y"))) : (mr[2], mr[3])
    depth = Float64.(rd("depth"))
    sig = Float64.(rd("sig"))
    dsig, sigf, dth = Float64.(rd("dsig")), Float64.(rd("sigf")), Float64(rd("dth")[1])
    fg = [Int(a):Int(b) for (a, b) in zip(rd("fgroup_first"), rd("fgroup_last"))]
    dg = [collect(Int(a):Int(a) + Int(l) - 1) for (a, l) in zip(rd("dgroup_first"), rd("dgroup_len"))]
    cases = [(Int(a), Int(b)) for (a, b) in zip(rd("case_fgroup"), rd("case_dgroup"))]
    nc = length(cases)
    done = Int.(rd("case_done")) .== 1                 # unwritten entries carry the NetCDF fill value
    kp_m0 = permutedims(Float32.(rd("kp_m0")), (3, 1, 2))
    hasdir = NetCDF.open(file) do nc_; haskey(nc_.vars, "kp_cos"); end       # files written before the direction moments: none
    kp_cos = hasdir ? permutedims(Float32.(rd("kp_cos")), (3, 1, 2)) : zeros(Float32, nc, 0, 0)
    kp_sin = hasdir ? permutedims(Float32.(rd("kp_sin")), (3, 1, 2)) : zeros(Float32, nc, 0, 0)
    haspts = NetCDF.open(file) do nc_; haskey(nc_.vars, "kp_spec"); end
    if haspts
        pts = [(Int(a), Int(b)) for (a, b) in zip(rd("point_i"), rd("point_j"))]
        kp_spec = permutedims(Float32.(rd("kp_spec")), (4, 1, 2, 3))
    else
        pts = Tuple{Int,Int}[]; kp_spec = zeros(Float32, nc, length(sig), round(Int, 2π / dth), 0)
    end
    ga = NetCDF.open(file) do nc_; Dict{String,Any}(nc_.gatts); end
    info = Dict{String,Any}("sides" => Symbol.(filter(!isempty, split(string(get(ga, "sides", "")), ","))),
                            "fgroup" => Int(get(ga, "fgroup", 1)), "dgroup" => Int(get(ga, "dgroup", 1)),
                            "seconds" => Float64(get(ga, "seconds", 0.0)), "refraction" => get(ga, "refraction", 1) != 0,
                            "breaking" => get(ga, "breaking", 0) != 0, "friction" => get(ga, "friction", 0) != 0,
                            "wind" => get(ga, "wind", 0) != 0)
    mr === nothing || (info["mesh"] = mr[1])
    Vin = Float64.(rd("Vin")); Vin[.!done] .= 0.0
    kp = BinWavesKP(x, y, depth, sig, dsig, sigf, dth, fg, dg, cases, Vin, kp_m0, kp_cos, kp_sin, pts, kp_spec, done, info)
    all(kp.done) || @warn "$file holds $(count(done))/$nc cases; reconstruct will warn until build_kp(...; checkpoint=\"$file\") has finished the rest"
    return kp
end

function _check_compatible(old::BinWavesKP, new::BinWavesKP, file)
    ok = length(old.x) == length(new.x) && length(old.y) == length(new.y) && length(old.sig) == length(new.sig) &&
         old.dth ≈ new.dth && old.sig ≈ new.sig && old.fgroups == new.fgroups && old.dgroups == new.dgroups &&
         old.cases == new.cases && old.points == new.points && old.x ≈ new.x && old.y ≈ new.y && old.depth ≈ new.depth &&
         string.(old.info["sides"]) == string.(new.info["sides"]) && _hasdir(old) == _hasdir(new)
    ok || error("checkpoint $file was made with a different grid, depth, spectral grid, grouping, points, boundary sides or " *
                "direction-moment setting (dir_moments); " *
                "delete it or pass resume = false")
end

# ------------------------------------------------------------------------------------------------------------
# slow part: run the cases
# ------------------------------------------------------------------------------------------------------------
"""
    build_kp(m; fgroup=1, dgroup=1, sides=(:left,:right,:bottom,:top), hs_rule=T->T>5 ? 1.0 : 0.1,
             points=[], sweep=SweepParams(), verbose=true, cases=nothing) -> BinWavesKP

Run one stationary solution per case on the model `m` (all its settings — depth, currents, physics flags — are used as
they are; the model's own boundary conditions are restored afterwards).

* `fgroup`, `dgroup`  merge this many neighbouring model frequency / direction bins into one case (fewer cases, coarser
  binning; the case spectrum is then flat in variance per model bin inside the group). `nth` must be a multiple of `dgroup`.
* `sweep`   settings of the stationary solver; `dabs` is applied per unit input height (scaled by each case's Hs).
            The default is **upwind in θ and σ** (`w_theta = w_sigma = 0`). Superposition needs a linear model, and the
            central-weighted default (0.5) rings around a narrow spectral bin; the solver clips the resulting negative action,
            which is non-linear (measured: 4.5 % error in superposed Hs with 0.5, 0.02 % with 0). The price is more numerical
            directional diffusion, so use a fine `nth` and run your direct comparisons with the same settings.
* `sides`   boundaries that carry the case spectrum; the others get `ZeroBC()`.
* `points`  where the full nearshore spectrum is stored: grid indices `(i, j)` or `(x, y)` coordinates (nearest cell). On a
            block mesh (`BlockModel`) give `(x, y)` coordinates; the fields are then vectors over the mesh cells (`(ncells, 1)`,
            see `cell_centres` and `to_grid`) and the mesh is kept in `kp.info["mesh"]`.
* `checkpoint`  path of a NetCDF file that is updated every `checkpoint_every` cases (and at the end). If it already exists and
            `resume = true` (default) the finished cases are read from it and skipped, so an interrupted build continues where
            it stopped. The file can be loaded with `load_kp_netcdf` at any time. Settings must match (error otherwise).
* `cases`   optional vector of case numbers to compute (for testing / restarts); other cases stay zero.
* `dir_moments`  also store, per case and cell, the energy-weighted direction vector (Σ E cos θ, Σ E sin θ) so that
            `reconstruct` gives the mean wave direction and directional spread at every cell (default; 3× the field storage
            of m0 alone). Like m0 it is linear in the spectrum, so the superposition is exact.
"""
function build_kp(m::Union{WaveModel,BlockModel}; fgroup::Int = 1, dgroup::Int = 1, sides = (:left, :right, :bottom, :top),
                  hs_rule = T -> T > 5 ? 1.0 : 0.1, points = Tuple{Int,Int}[], sweep = SweepParams(w_theta = 0.0, w_sigma = 0.0), verbose::Bool = true,
                  cases = nothing, checkpoint = nothing, checkpoint_every::Int = 10, resume::Bool = true,
                  dir_moments::Bool = true)
    sp = m.spec
    ns, nth = sp.ns, sp.nth
    nx, ny = _fieldsize(m)
    nth % dgroup == 0 || error("nth = $nth is not a multiple of dgroup = $dgroup")
    fgroups = [i:min(i + fgroup - 1, ns) for i in 1:fgroup:ns]
    dgroups = [collect(h*dgroup+1:(h+1)*dgroup) for h in 0:(nth÷dgroup-1)]
    caselist = [(a, b) for a in eachindex(fgroups) for b in eachindex(dgroups)]
    nc = length(caselist)
    run_these = cases === nothing ? (1:nc) : cases
    points = _resolve_points(m, points)
    npt = length(points)
    Vin = zeros(nc)
    kp_m0 = zeros(Float32, nc, nx, ny)
    kp_cos = dir_moments ? zeros(Float32, nc, nx, ny) : zeros(Float32, nc, 0, 0)
    kp_sin = dir_moments ? zeros(Float32, nc, nx, ny) : zeros(Float32, nc, 0, 0)
    kp_spec = zeros(Float32, nc, ns, nth, npt)
    done = fill(false, nc)
    orig = m.bc
    D = _depthfield(m)
    info = Dict{String,Any}("sides" => collect(sides), "fgroup" => fgroup, "dgroup" => dgroup, "seconds" => 0.0,
                            "refraction" => m.params.refraction, "breaking" => m.params.breaking, "friction" => m.params.friction,
                            "wind" => m.params.wind)
    _meshinfo!(info, m)
    xg, yg = _coords(m)
    kp = BinWavesKP(xg, yg, copy(D), copy(sp.sig), copy(sp.dsig), copy(sp.sigf), sp.dth, fgroups, dgroups, caselist,
                    Vin, kp_m0, kp_cos, kp_sin, collect(points), kp_spec, done, info)
    t_prev = 0.0
    if checkpoint !== nothing
        if resume && isfile(checkpoint)
            old = load_kp_netcdf(checkpoint)
            _check_compatible(old, kp, checkpoint)
            Vin .= old.Vin; kp_m0 .= old.kp_m0; kp_cos .= old.kp_cos; kp_sin .= old.kp_sin; kp_spec .= old.kp_spec; done .= old.done
            t_prev = Float64(get(old.info, "seconds", 0.0))
            verbose && @printf("  resuming from %s: %d/%d cases already done\n", checkpoint, count(done), nc)
        else
            _create_kp_nc(checkpoint, kp)
        end
    end
    pending = Int[]
    flush_ck!() = begin
        if checkpoint !== nothing && !isempty(pending)
            _write_cases!(checkpoint, kp, pending)
            ncputatt(checkpoint, "NC_GLOBAL", Dict{String,Any}("seconds" => t_prev + (time() - t0)))
            empty!(pending)
        end
    end
    t0 = time()
    todo = [c for c in run_these if !done[c]]
    for (n, c) in enumerate(todo)
        a, b = caselist[c]
        fr, dr = fgroups[a], dgroups[b]
        fc = sp.sig[fr[(1 + end) ÷ 2]] / 2π                          # representative frequency of the group
        Hs_in = hs_rule(1 / fc)
        V = Hs_in^2 / 16
        Vin[c] = V
        vbin = V / (length(fr) * length(dr))
        Nc = zeros(ns, nth)
        for is in fr, it in dr
            Nc[is, it] = vbin / (sp.sig[is] * sp.dsig[is] * sp.dth)      # E = σ N, variance = E Δσ Δθ
        end
        initialize!(m; Hs = 0.0)
        bc(s) = s in sides ? SpectrumBC(Nc) : ZeroBC()
        set_boundaries!(m; left = bc(:left), right = bc(:right), bottom = bc(:bottom), top = bc(:top))
        # the SWAN-like absolute tolerance (dabs, in metres) must scale with the case's own height: a 0.1 m case with a 5 mm
        # tolerance would otherwise stop after a couple of sweeps with several % error
        swc = SweepParams(w_theta = sweep.w_theta, w_sigma = sweep.w_sigma, iters = sweep.iters, maxiter = sweep.maxiter,
                          drel = sweep.drel, dabs = sweep.dabs * Hs_in, npnts = sweep.npnts)
        solve_stationary!(m; sweep = swc, verbose = false)
        if dir_moments
            _moments!(view(kp_m0, c, :, :), view(kp_cos, c, :, :), view(kp_sin, c, :, :), m, 1 / V)
        else
            _moments!(view(kp_m0, c, :, :), nothing, nothing, m, 1 / V)
        end
        for (p, pt) in enumerate(points)
            _binvar!(view(kp_spec, c, :, :, p), m, pt, 1 / V)
        end
        done[c] = true
        push!(pending, c)
        length(pending) >= checkpoint_every && flush_ck!()
        if verbose && (n == 1 || n % max(1, length(todo) ÷ 20) == 0 || n == length(todo))
            el = time() - t0
            @printf("  case %d/%d (f = %.3f Hz, θ-group %d)   %.1f s elapsed, ~%.0f s remaining\n", n, length(todo), fc, b, el,
                    el / n * (length(todo) - n))
        end
    end
    flush_ck!()
    set_boundaries!(m; left = orig[1], right = orig[2], bottom = orig[3], top = orig[4])
    info["seconds"] = t_prev + (time() - t0)
    checkpoint === nothing || ncputatt(checkpoint, "NC_GLOBAL", Dict{String,Any}("seconds" => info["seconds"]))
    return kp
end

# ------------------------------------------------------------------------------------------------------------
# fast part: bin the offshore spectrum and superpose
# ------------------------------------------------------------------------------------------------------------
function _edges(c::AbstractVector)
    n = length(c); e = zeros(n + 1)
    n == 1 && return [c[1] - 0.5, c[1] + 0.5]
    for k in 2:n; e[k] = (c[k-1] + c[k]) / 2; end
    e[1] = c[1] - (c[2] - c[1]) / 2; e[n+1] = c[n] + (c[n] - c[n-1]) / 2
    return e
end
_overlap(a, b, c, d) = max(0.0, min(b, d) - max(a, c))
function _circ_overlap(a, b, c, d)                    # intervals on a 360° circle
    s = 0.0
    for k in -2:2; s += _overlap(a, b, c + 360k, d + 360k); end
    return s
end

"""
    bin_offshore_spectrum(kp, E, f, dir; units = :m2_Hz_deg) -> (V (nt × ncase), captured (nt))

Bin an offshore variance-density spectrum `E[t, f, dir]` (m²/Hz/deg; frequency `f` in Hz ascending, `dir` in degrees,
nautical *coming-from*, uniformly spaced) onto the case bins by exact overlap of the bin areas. `captured` is the fraction of the
offshore variance that falls inside the model's spectral range (energy outside [fmin, fmax] cannot be propagated).
"""
function bin_offshore_spectrum(kp::BinWavesKP, E::AbstractArray{<:Real,3}, f::AbstractVector, dir::AbstractVector)
    nt, nf, nd = size(E)
    length(f) == nf && length(dir) == nd || error("E must be (time, frequency, direction)")
    issorted(f) || error("frequencies must be ascending")
    ef = _edges(collect(Float64, f)); df = diff(ef)
    dd = nd > 1 ? median_spacing(dir) : 360.0
    ng, nh = length(kp.fgroups), length(kp.dgroups)
    Af = zeros(ng, nf)                                                         # overlap length (Hz)
    for (a, fr) in enumerate(kp.fgroups)
        lo, hi = kp.sigf[first(fr)] / 2π, kp.sigf[last(fr)+1] / 2π
        for i in 1:nf; Af[a, i] = _overlap(lo, hi, ef[i], ef[i+1]); end
    end
    dthd = rad2deg(kp.dth)
    Ad = zeros(nh, nd)                                                         # overlap length (deg)
    for (h, mem) in enumerate(kp.dgroups)
        θlo = (first(mem) - 1 - 0.5) * dthd; θhi = θlo + length(mem) * dthd    # math, propagating-to
        lo, hi = 270 - θhi, 270 - θlo                                          # nautical, coming-from
        for j in 1:nd; Ad[h, j] = _circ_overlap(lo, hi, dir[j] - dd / 2, dir[j] + dd / 2); end
    end
    V = zeros(nt, length(kp.cases)); captured = zeros(nt)
    tot = zeros(nt)
    Et = zeros(nf, nd)
    for t in 1:nt
        Et .= @view E[t, :, :]
        tot[t] = sum(Et[i, j] * df[i] * dd for i in 1:nf, j in 1:nd)
        W = Af * Et * Ad'                                                      # (ng × nh) variance m²
        for (c, (a, h)) in enumerate(kp.cases); V[t, c] = W[a, h]; end
        captured[t] = tot[t] > 0 ? sum(@view V[t, :]) / tot[t] : 1.0
    end
    return V, captured
end
median_spacing(d) = (s = sort(collect(Float64, d)); sort(diff(s))[(1 + length(s) - 1) ÷ 2])

"""
    reconstruct(kp, E, f, dir; fields = true, wind = nothing, U10 = nothing, wdir = nothing)
        -> (; Hs, Dir, Spread, point_var, V, captured[, wind])

Nearshore reconstruction for offshore spectra `E[t, f, dir]` (see `bin_offshore_spectrum`).
* `Hs`         (nx, ny, nt) significant wave height (Float32; 0 on dry cells) — only if `fields = true`; on a block mesh
               (ncells, 1, nt): `to_grid(kp, rec.Hs, xg, yg)` samples it onto a regular grid
* `Dir`        (nx, ny, nt) mean wave direction (nautical, coming from, degrees; NaN without energy): the direction of the
               energy-weighted vector (Σ E cos θ, Σ E sin θ), as `bulk` defines it. Needs coefficients built with
               `dir_moments = true` (the default); otherwise `Dir` and `Spread` are empty (0, 0, 0) arrays.
* `Spread`     (nx, ny, nt) directional spread (degrees): √(2(1 − r)), r = |(Σ E cos θ, Σ E sin θ)| / m0 (Kuik et al. 1988)
* `point_var`  (nt, ns, nth, npoints) variance (m²) per model spectral bin at the stored points
* `V`          the case weights, `captured` the in-range fraction of the offshore variance
* `depth_limit`  e.g. `0.73`: cap the gridded Hs at `depth_limit · depth` (use with coefficients built *without* breaking, see README)

Local wind sea (semi-enclosed water): pass a `WindKP` from `build_wind_kp` as `wind`, with the local wind time series
`U10[t]` (m/s) and `wdir[t]` (nautical, coming from, degrees). Its variance is added to the swell (`Hs² = Hs²_swell +
Hs²_wind`, and the point spectra are summed); `wind_interp` (`:loglog` / `:linear`) and `umin` are passed to `wind_sea`. The
result then also holds `wind`, the separate wind-sea part (see `wind_sea`).
"""
function reconstruct(kp::BinWavesKP, E::AbstractArray{<:Real,3}, f::AbstractVector, dir::AbstractVector; fields::Bool = true,
                     tchunk::Int = 128, depth_limit = nothing, wind = nothing, U10 = nothing, wdir = nothing,
                     wind_interp::Symbol = :loglog, umin::Real = 1.0)
    all(kp.done) || @warn "only $(count(kp.done)) of $(length(kp.cases)) cases are computed: the reconstructed energy is incomplete"
    V, captured = bin_offshore_spectrum(kp, E, f, dir)
    nt = size(V, 1); nc = length(kp.cases)
    nx, ny = size(kp.kp_m0, 2), size(kp.kp_m0, 3)
    ws = nothing
    if wind !== nothing
        wind isa WindKP || error("wind must be a WindKP (from build_wind_kp / load_wind_kp_netcdf)")
        (U10 === nothing || wdir === nothing) && error("give the local wind time series U10 and wdir with wind")
        length(U10) == nt && length(wdir) == nt || error("U10 and wdir must have one value per time of E ($nt)")
        (size(wind.m0, 3) == nx && size(wind.m0, 4) == ny && wind.x ≈ kp.x && wind.y ≈ kp.y && wind.sig ≈ kp.sig &&
         wind.dth ≈ kp.dth) || error("the wind table was built on a different grid / spectral grid than the swell coefficients")
        wind.points == kp.points || error("the wind table and the swell coefficients store spectra at different points")
        ws = wind_sea(wind, U10, wdir; fields = fields, interp = wind_interp, umin = umin)
    end
    dodir = fields && _hasdir(kp)
    if dodir && ws !== nothing && size(ws.mc, 1) == 0
        @warn "the wind table has no direction moments (built before they were added): Dir and Spread are not reconstructed"
        dodir = false
    end
    Hs = fields ? zeros(Float32, nx, ny, nt) : zeros(Float32, 0, 0, 0)
    Dir = dodir ? zeros(Float32, nx, ny, nt) : zeros(Float32, 0, 0, 0)
    Spread = dodir ? zeros(Float32, nx, ny, nt) : zeros(Float32, 0, 0, 0)
    if fields
        K = reshape(kp.kp_m0, nc, nx * ny)
        Kc = dodir ? reshape(kp.kp_cos, nc, nx * ny) : K
        Ks = dodir ? reshape(kp.kp_sin, nc, nx * ny) : K
        for t0 in 1:tchunk:nt
            r = t0:min(nt, t0 + tchunk - 1)
            Vr = Float32.(V[r, :])
            m0 = Vr * K                                                        # (chunk × cells)
            ws === nothing || (m0 .+= permutedims(reshape(ws.m0[:, :, r], nx * ny, length(r))))
            Hs[:, :, r] .= reshape(permutedims(4 .* sqrt.(max.(m0, 0f0))), nx, ny, length(r))
            if dodir
                A = Vr * Kc; B = Vr * Ks
                if ws !== nothing
                    A .+= permutedims(reshape(ws.mc[:, :, r], nx * ny, length(r)))
                    B .+= permutedims(reshape(ws.ms[:, :, r], nx * ny, length(r)))
                end
                for (q, k) in enumerate(r), c in 1:nx * ny
                    d, sp_ = _dir_spread(m0[q, c], A[q, c], B[q, c])
                    Dir[c + (k - 1) * nx * ny] = d; Spread[c + (k - 1) * nx * ny] = sp_
                end
            end
            if depth_limit !== nothing            # crude surf-zone cap: Hs ≤ γ d
                cap = Float32.(depth_limit .* kp.depth)
                for k in r; @views Hs[:, :, k] .= min.(Hs[:, :, k], cap); end
            end
        end
    end
    ns, nth, np = size(kp.kp_spec, 2), size(kp.kp_spec, 3), size(kp.kp_spec, 4)
    pv = np == 0 ? zeros(nt, ns, nth, 0) : reshape(V * reshape(Float64.(kp.kp_spec), nc, ns * nth * np), nt, ns, nth, np)
    ws === nothing && return (; Hs, Dir, Spread, point_var = pv, V, captured)
    pv .+= ws.point_var
    return (; Hs, Dir, Spread, point_var = pv, V, captured, wind = ws)
end

"""
    point_parameters(kp, point_var) -> (; Hs, Tm01, Tp, Dir)   each (nt, npoints)   (kp: BinWavesKP or WindKP)

Bulk parameters from reconstructed point spectra: Hs = 4√m0, Tm01 = m0/m1, Tp = 1/f at the maximum of the direction-integrated
density, Dir = vector-mean coming-from direction (nautical, degrees).
"""
function point_parameters(kp, pv::AbstractArray{<:Real,4})   # kp: BinWavesKP or WindKP
    nt, ns, nth, np = size(pv)
    sig, dsig, dth = kp.sig, kp.dsig, kp.dth
    f = sig ./ 2π
    Hs = zeros(nt, np); Tm = zeros(nt, np); Tp = zeros(nt, np); Dir = zeros(nt, np)
    cth = cos.((0:nth-1) .* dth); sth = sin.((0:nth-1) .* dth)
    for p in 1:np, t in 1:nt
        m0 = 0.0; m1 = 0.0; sx = 0.0; sy = 0.0; best = -1.0; ipk = 1
        for is in 1:ns
            vf = 0.0
            for it in 1:nth
                v = pv[t, is, it, p]; vf += v; sx += v * cth[it]; sy += v * sth[it]
            end
            m0 += vf; m1 += vf * f[is]
            dens = vf / dsig[is]
            dens > best && (best = dens; ipk = is)
        end
        Hs[t, p] = 4 * sqrt(max(m0, 0.0))
        Tm[t, p] = m0 > 0 ? m0 / m1 : NaN
        Tp[t, p] = m0 > 0 ? 1 / f[ipk] : NaN
        Dir[t, p] = m0 > 0 ? mod(270 - rad2deg(atan(sy, sx)), 360) : NaN
    end
    return (; Hs, Tm01 = Tm, Tp, Dir)
end

"""
    to_grid(kp, v, xg, yg; outside = NaN)

Sample a field of coefficients built on a block mesh (e.g. `reconstruct(...).Hs`, (ncells, 1, nt)) onto the regular grid
`xg × yg`. For coefficients on a regular grid, `v` is returned unchanged.
"""
function WaveBlocks.to_grid(kp::BinWavesKP, v::AbstractArray, xg::AbstractVector, yg::AbstractVector; outside = NaN)
    _hasmesh(kp.info) || return v
    return to_grid(kp.info["mesh"], v, xg, yg; outside = outside)
end
