# CF-compliant NetCDF output of bulk wave fields (block meshes and regular grids). Included by src/BinWaves.jl.

# CF metadata of the known fields (standard names from the CF standard name table)
const _CF = Dict{Symbol,Dict{String,Any}}(
    :Hs  => Dict("standard_name" => "sea_surface_wave_significant_height", "long_name" => "significant wave height (4 sqrt m0)", "units" => "m"),
    :Tm  => Dict("standard_name" => "sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment",
                 "long_name" => "mean wave period Tm01", "units" => "s"),
    :Tm01 => Dict("standard_name" => "sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment",
                 "long_name" => "mean wave period Tm01", "units" => "s"),
    :Tp  => Dict("standard_name" => "sea_surface_wave_period_at_variance_spectral_density_maximum", "long_name" => "peak wave period", "units" => "s"),
    :Dir => Dict("standard_name" => "sea_surface_wave_from_direction", "long_name" => "mean wave direction (coming from, clockwise from north)",
                 "units" => "degree"),
    :Sxx => Dict("long_name" => "radiation stress, xx component", "units" => "N m-1"),
    :Sxy => Dict("long_name" => "radiation stress, xy component", "units" => "N m-1"),
    :Syy => Dict("long_name" => "radiation stress, yy component", "units" => "N m-1"),
    :depth => Dict("standard_name" => "sea_floor_depth_below_sea_surface", "long_name" => "still water depth (positive down)", "units" => "m"),
)
const _CIRCULAR = Set([:Dir])            # interpolated through their unit vector

"""
    BulkWriter(file, model_or_mesh; layout = :levels, vars = (:Hs, :Tm, :Dir), interp = :bilinear,
               time_units = "seconds since 1970-01-01 00:00:00", crs = nothing, title = "...", attributes = Dict(),
               global_attributes = Dict(), compress = 4, T = Float32, hmin = 0.05)

Open a CF-1.8 NetCDF file for bulk wave fields, one record per call to `write_bulk!` / `write_fields!` (unlimited `time`).

`model_or_mesh`: a `BlockModel`, a `BlockMesh` (then pass `depth`, a vector over the cells, to get dry-cell masking and the
depth variable), a `BinWavesKP` / `WindKP` built on a block mesh (for `reconstruct` output), or a regular-grid `WaveModel`.

Layouts (block meshes; a regular grid always gives a single (x, y) layer):
* `:levels`  native storage: one regular grid per refinement level present, covering the whole domain at that level's cell
             size (`x_L0`, `y_L0`, …), with variables `Hs_L0`, `Hs_L1`, … The values sit in the layer of the level they
             were computed at and are NaN (`_FillValue`) everywhere else, so every cell appears in exactly one layer.
* `:finest`  all fields interpolated onto the grid of the finest level present (`x`, `y`): `interp = :bilinear` (between
             cell centres, exact at the finest cells, dry cells excluded) or `:nearest` (the containing cell's value). A
             `level` variable gives the refinement level each point comes from.
* `:both`    both in the same file.
Dry cells are NaN; `depth` (and `level`) are written once, without time.

`vars`: fields to store, from `bulk` (`:Hs, :Tm, :Dir, :Sxx, :Sxy, :Syy`) or any names you pass to `write_fields!`.
Known names get CF standard names and units; set or override attributes with `attributes = Dict(:name => Dict(...))`.
`crs`: a Dict of CF grid-mapping attributes (e.g. `"grid_mapping_name" => "transverse_mercator", ...`, and/or `"crs_wkt"`);
it becomes the `crs` variable referenced by every field's `grid_mapping`. x and y are projection coordinates in metres.
"""
mutable struct BulkWriter
    nc::NetCDF.NcFile
    file::String
    layout::Symbol
    vars::Vector{Symbol}
    mesh::Union{Nothing,BlockMesh}
    nfield::Int                                  # number of cells (block mesh) or nx·ny (grid)
    wet::Vector{Bool}
    levels::Vector{Int}
    lev_cells::Dict{Int,Vector{Int}}             # cells at level l
    lev_pos::Dict{Int,Vector{Int}}               # their linear position in the level-l grid
    lev_n::Dict{Int,Int}                         # size of the level-l grid
    stencil::Union{Nothing,GridStencil}
    T::DataType
    nt::Int
    isopen::Bool
end
Base.show(io::IO, w::BulkWriter) = print(io, "BulkWriter(", w.file, ", layout = :", w.layout, ", vars = ", w.vars, ", ", w.nt, " records",
    w.isopen ? "" : ", closed", ")")

_level_tag(l) = "L$l"

function BulkWriter(file::AbstractString, src; layout::Symbol = :levels, vars = (:Hs, :Tm, :Dir), interp::Symbol = :bilinear,
                    time_units::AbstractString = "seconds since 1970-01-01 00:00:00", crs = nothing,
                    title::AbstractString = "BinWaves.jl wave model output", attributes = Dict{Symbol,Dict{String,Any}}(),
                    global_attributes = Dict{String,Any}(), compress::Integer = 4, T::DataType = Float32, hmin = 0.05, depth = nothing)
    layout in (:levels, :finest, :both) || error("layout must be :levels, :finest or :both")
    T in (Float32, Float64) || error("T must be Float32 or Float64")
    mesh, dep, xg, yg = _output_geometry(src, depth)
    dep = dep === nothing ? nothing : Float64.(vec(dep))
    nfield = mesh === nothing ? length(xg) * length(yg) : ncells(mesh)
    wet = dep === nothing ? fill(true, nfield) : dep .> hmin
    vars = Symbol.(collect(vars))
    isfile(file) && rm(file)
    FV = Dict{String,Any}("_FillValue" => T(NaN))
    tdim = NetCDF.NcDim("time", 0; unlimited = true)
    tvar = NetCDF.NcVar("time", tdim; t = Float64, atts = Dict{String,Any}("standard_name" => "time", "long_name" => "time",
                        "units" => time_units, "calendar" => "standard", "axis" => "T"))
    xyatts(ax, l) = Dict{String,Any}("standard_name" => "projection_$(ax)_coordinate", "units" => "m", "axis" => uppercase(ax),
        "long_name" => "$ax coordinate of the cell centres" * (l === nothing ? "" : " (refinement level $l, cell size $(block_dx(mesh, l)) m)"))
    varatts(v) = merge(get(_CF, v, Dict{String,Any}("long_name" => string(v), "units" => "1")), get(attributes, v, Dict{String,Any}()))
    gm = crs === nothing ? Dict{String,Any}() : Dict{String,Any}("grid_mapping" => "crs")
    ncvars = NetCDF.NcVar[tvar]
    levels = Int[]; lev_cells = Dict{Int,Vector{Int}}(); lev_pos = Dict{Int,Vector{Int}}(); lev_n = Dict{Int,Int}()
    stencil = nothing
    static = Tuple{String,Any}[]                       # (variable, data) written after creation
    if mesh === nothing                                # regular grid: one layer
        xd = NetCDF.NcDim("x", xg, xyatts("x", nothing)); yd = NetCDF.NcDim("y", yg, xyatts("y", nothing))
        for v in vars
            push!(ncvars, NetCDF.NcVar(string(v), [xd, yd, tdim]; t = T, atts = merge(varatts(v), FV, gm), compress = compress))
        end
        if dep !== nothing
            push!(ncvars, NetCDF.NcVar("depth", [xd, yd]; t = T, atts = merge(varatts(:depth), FV, gm), compress = compress))
            push!(static, ("depth", reshape(T.(ifelse.(wet, dep, NaN)), length(xg), length(yg))))
        end
    else
        if layout in (:levels, :both)
            levels = sort(unique(mesh.level))
            x, y = cell_centres(mesh)
            for l in levels
                lx, ly = level_grid(mesh, l); h = block_dx(mesh, l)
                cells = [c for k in 1:nblocks(mesh) if mesh.level[k] == l for c in (k - 1) * mesh.bs^2 + 1:k * mesh.bs^2]
                pos = [round(Int, (x[c] - mesh.x0) / h + 0.5) + (round(Int, (y[c] - mesh.y0) / h + 0.5) - 1) * length(lx) for c in cells]
                lev_cells[l] = cells; lev_pos[l] = pos; lev_n[l] = length(lx) * length(ly)
                tag = _level_tag(l)
                xd = NetCDF.NcDim("x_$tag", lx, xyatts("x", l)); yd = NetCDF.NcDim("y_$tag", ly, xyatts("y", l))
                la = Dict{String,Any}("refinement_level" => Int32(l), "cell_size" => h,
                    "comment" => "values of the cells at refinement level $l; NaN where the mesh is at another level or the cell is dry")
                for v in vars
                    a = merge(varatts(v), FV, gm, la); a["long_name"] = a["long_name"] * " (level $l)"
                    push!(ncvars, NetCDF.NcVar("$(v)_$tag", [xd, yd, tdim]; t = T, atts = a, compress = compress))
                end
                if dep !== nothing
                    a = merge(varatts(:depth), FV, gm, la); a["long_name"] = a["long_name"] * " (level $l)"
                    push!(ncvars, NetCDF.NcVar("depth_$tag", [xd, yd]; t = T, atts = a, compress = compress))
                    D = fill(T(NaN), lev_n[l]); D[pos] .= T.(ifelse.(wet[cells], dep[cells], NaN))
                    push!(static, ("depth_$tag", reshape(D, length(lx), length(ly))))
                end
            end
        end
        if layout in (:finest, :both)
            fx, fy = finest_grid(mesh)
            stencil = interp_stencil(mesh, fx, fy; method = interp)
            lf = maximum(mesh.level)
            xd = NetCDF.NcDim("x", fx, xyatts("x", lf)); yd = NetCDF.NcDim("y", fy, xyatts("y", lf))
            ia = Dict{String,Any}("comment" => "interpolated onto the grid of the finest refinement level ($(block_dx(mesh, lf)) m) " *
                                  (interp === :bilinear ? "by bilinear interpolation between cell centres (dry cells excluded)" :
                                                          "by taking the value of the containing cell"))
            for v in vars
                push!(ncvars, NetCDF.NcVar(string(v), [xd, yd, tdim]; t = T, atts = merge(varatts(v), FV, gm, ia), compress = compress))
            end
            push!(ncvars, NetCDF.NcVar("level", [xd, yd]; t = Int32, compress = compress,
                  atts = merge(Dict{String,Any}("long_name" => "refinement level of the mesh cell containing the point", "units" => "1",
                               "_FillValue" => Int32(-1)), gm)))
            push!(static, ("level", reshape(Int32[c == 0 ? -1 : mesh.level[(c - 1) ÷ mesh.bs^2 + 1] for c in stencil.own], length(fx), length(fy))))
            if dep !== nothing
                push!(ncvars, NetCDF.NcVar("depth", [xd, yd]; t = T, atts = merge(varatts(:depth), FV, gm, ia), compress = compress))
                push!(static, ("depth", reshape(T.(to_grid(ifelse.(wet, dep, NaN), stencil)), length(fx), length(fy))))
            end
        end
        # the block list, so that the mesh can be rebuilt from the file
        bd = NetCDF.NcDim("block", nblocks(mesh))
        for (nm, ln, data) in (("mesh_level", "refinement level of each block", mesh.level), ("mesh_bx", "block x index at its level", mesh.bx),
                                ("mesh_by", "block y index at its level", mesh.by))
            push!(ncvars, NetCDF.NcVar(nm, bd; t = Int32, atts = Dict{String,Any}("long_name" => ln, "units" => "1")))
            push!(static, (nm, Int32.(data)))
        end
    end
    if crs !== nothing
        ca = Dict{String,Any}(string(k) => v for (k, v) in crs)
        haskey(ca, "grid_mapping_name") || haskey(ca, "crs_wkt") || error("crs needs grid_mapping_name (CF) and/or crs_wkt")
        push!(ncvars, NetCDF.NcVar("crs", NetCDF.NcDim[]; t = Int32, atts = ca))
    end
    ga = Dict{String,Any}("Conventions" => "CF-1.8", "title" => title, "source" => "BinWaves.jl",
                          "history" => "created " * Dates_now() * " by BinWaves.jl", "storage_layout" => string(layout))
    if mesh !== nothing
        merge!(ga, Dict{String,Any}("mesh_x0" => mesh.x0, "mesh_y0" => mesh.y0, "mesh_dx0" => mesh.dx0, "mesh_bs" => Int32(mesh.bs),
            "mesh_nbx" => Int32(mesh.nbx), "mesh_nby" => Int32(mesh.nby), "mesh_maxlevel" => Int32(mesh.maxlevel),
            "mesh_note" => "block-adaptive quadtree mesh (BinWaves.WaveBlocks): blocks of mesh_bs x mesh_bs cells; level l has cells of " *
                           "mesh_dx0 / 2^l; block k covers [x0 + bx*bs*dx, x0 + (bx+1)*bs*dx] (and the same in y)"))
        layout in (:levels, :both) && (ga["refinement_levels"] = Int32.(levels))
    end
    merge!(ga, Dict{String,Any}(string(k) => v for (k, v) in global_attributes))
    nc = NetCDF.create(file, ncvars; gatts = ga, mode = NetCDF.NC_NETCDF4, add_finalizer = false)
    for (nm, data) in static
        NetCDF.putvar(nc[nm], data)
    end
    NetCDF.sync(nc)
    w = BulkWriter(nc, String(file), layout, vars, mesh, nfield, wet, levels, lev_cells, lev_pos, lev_n, stencil, T, 0, true)
    finalizer(close, w)                   # a writer that is never closed is closed when garbage collected
    return w
end

_output_geometry(bm::BlockModel, depth) = (bm.mesh, depth === nothing ? cell_depth(bm) : depth, nothing, nothing)
_output_geometry(m::BlockMesh, depth) = (m, depth, nothing, nothing)
_output_geometry(m::WaveModel, depth) = (nothing, depth === nothing ? _depthfield(m) : depth, m.grid.x, m.grid.y)
function _output_geometry(k::Union{BinWavesKP,WindKP}, depth)
    d = depth === nothing ? k.depth : depth
    _hasmesh(k.info) ? (k.info["mesh"], d, nothing, nothing) : (nothing, d, k.x, k.y)
end

"""
    write_fields!(w, t; Hs = v, Tm = v2, ...)

Append one time record (`t` in the writer's `time_units`) with cell fields (vectors over the cells of the mesh, or (nx, ny)
matrices on a regular grid). Every name declared in `vars` must be given. NaN marks missing values; dry cells are set to NaN.
"""
function write_fields!(w::BulkWriter, t::Real; fields...)
    w.isopen || error("the writer is closed")
    for v in w.vars
        haskey(fields, v) || error("field $v is declared in vars but not given")
    end
    n = w.nt + 1
    NetCDF.putvar(w.nc["time"], [Float64(t)]; start = [n], count = [1])
    for v in w.vars
        val = Float64.(vec(fields[v]))
        length(val) == w.nfield || error("field $v has $(length(val)) values, expected $(w.nfield)")
        val[.!w.wet] .= NaN
        if w.mesh === nothing
            nx, ny = size(w.nc[string(v)])[1:2]
            NetCDF.putvar(w.nc[string(v)], reshape(w.T.(val), nx, ny, 1); start = [1, 1, n], count = [nx, ny, 1])
            continue
        end
        for l in w.levels
            L = fill(NaN, w.lev_n[l]); L[w.lev_pos[l]] .= val[w.lev_cells[l]]
            var = w.nc["$(v)_$(_level_tag(l))"]
            nx, ny = size(var)[1:2]
            NetCDF.putvar(var, reshape(w.T.(L), nx, ny, 1); start = [1, 1, n], count = [nx, ny, 1])
        end
        if w.stencil !== nothing
            st = w.stencil
            G = if v in _CIRCULAR                      # through the unit vector (no averaging across 0/360)
                c = to_grid(cosd.(val), st); s = to_grid(sind.(val), st)
                mod.(atand.(s, c), 360.0)
            else
                to_grid(val, st)
            end
            NetCDF.putvar(w.nc[string(v)], reshape(w.T.(G), st.nx, st.ny, 1); start = [1, 1, n], count = [st.nx, st.ny, 1])
        end
    end
    w.nt = n
    NetCDF.sync(w.nc)
    return w
end

"""
    write_bulk!(w, m; t = m.t)

Append the bulk fields of a `BlockModel` or `WaveModel` (`bulk(m)`) at time `t`. Mean period and direction are NaN where
there is no wave energy (Hs < 1 mm). Use it as a `run!` callback: `run!(bm; tend, output_dt, callback = (m, t) -> write_bulk!(w, m))`.
"""
function write_bulk!(w::BulkWriter, m::Union{BlockModel,WaveModel}; t::Real = m.t)
    b = bulk(m)
    calm = vec(b.Hs) .< 1e-3
    f = Dict{Symbol,Any}()
    for v in w.vars
        hasproperty(b, v) || error("$v is not a bulk() field; use write_fields! for other variables")
        x = Float64.(vec(getproperty(b, v)))
        v in (:Tm, :Dir) && (x[calm] .= NaN)
        f[v] = x
    end
    return write_fields!(w, t; f...)
end

"""Close the file (also done when the writer is garbage collected). The file is synced after every record."""
Base.close(w::BulkWriter) = (w.isopen && (w.isopen = false; NetCDF._close(w.nc)); nothing)

"""
    save_bulk_netcdf(file, m; t = m.t, kwargs...)

Write the bulk fields of a `BlockModel` / `WaveModel` (one record) to a CF NetCDF file; keyword arguments as `BulkWriter`.
"""
function save_bulk_netcdf(file::AbstractString, m::Union{BlockModel,WaveModel}; t::Real = m.t, kwargs...)
    w = BulkWriter(file, m; kwargs...)
    try
        write_bulk!(w, m; t = t)
    finally
        close(w)
    end
    return file
end

"""
    save_fields_netcdf(file, src, times; fields..., kwargs...)

Write time series of cell fields, e.g. the output of `reconstruct` (`rec.Hs`, (ncells, 1, nt) on a block mesh or (nx, ny, nt)),
to a CF NetCDF file: `save_fields_netcdf("out.nc", kp, t; Hs = rec.Hs, layout = :both)`. `src` is anything `BulkWriter`
takes (the `BinWavesKP` gives the mesh and the depth). Each field's last dimension is time (length(times)).
"""
function save_fields_netcdf(file::AbstractString, src, times::AbstractVector; layout::Symbol = :levels, interp::Symbol = :bilinear,
                            time_units::AbstractString = "seconds since 1970-01-01 00:00:00", crs = nothing,
                            title::AbstractString = "BinWaves.jl wave model output", attributes = Dict{Symbol,Dict{String,Any}}(),
                            global_attributes = Dict{String,Any}(), compress::Integer = 4, T::DataType = Float32, hmin = 0.05,
                            depth = nothing, fields...)
    isempty(fields) && error("give at least one field, e.g. Hs = rec.Hs")
    w = BulkWriter(file, src; layout, vars = collect(keys(fields)), interp, time_units, crs, title, attributes, global_attributes,
                   compress, T, hmin, depth)
    try
        nt = length(times)
        for (n, t) in enumerate(times)
            f = Dict{Symbol,Any}()
            for (k, v) in pairs(fields)
                V = (ndims(v) == 1 && nt == 1) ? reshape(v, :, 1) : reshape(v, :, size(v)[end])
                size(V, 2) == nt || error("field $k: last dimension must be time ($nt)")
                f[k] = view(V, :, n)
            end
            write_fields!(w, t; f...)
        end
    finally
        close(w)
    end
    return file
end
