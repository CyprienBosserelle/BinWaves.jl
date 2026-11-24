"""
    WaveBlocks

Block-adaptive (quadtree) meshes for the WaveAction / WaveSweep wave models: coarse cells in deep water, fine cells where
the bathymetry is shallow and complex, at a fraction of the cost of a uniformly fine grid.

The domain is covered by square **blocks** of `bs × bs` cells (16 by default). A block at refinement level `l` has cells of
size `dx0 / 2^l`; refining a block replaces it by its four children. Face-neighbouring blocks differ by at most one level
(2:1 balance). `refine_by_depth!` refines from the bathymetry: a block is split while its cell size is larger than
`resolution(d)` at the shallowest wet depth `d` found inside it, so coastlines and shoals get the finest cells.

Each block is an ordinary `WaveModel` on its own uniform grid with two ghost layers; the blocks are coupled through their
ghost cells, filled from the neighbours: a copy at the same level, the coarse cell on the fine side (injection), the average
of the four fine cells on the coarse side (restriction). All the cell kernels (transport, refraction, sources, sweeps, GSE)
are those of the uniform-grid solvers, unchanged.

Solvers
* `solve_stationary!(bm)`: stationary four-sweep solver. In each sweep the blocks are visited in the sweep's upwind order
  (Gauss–Seidel across blocks); blocks that do not depend on each other are processed in parallel (threads).
* `run_implicit!(bm)` / `step_implicit!`: implicit non-stationary solver, same sweeps with a time term.
* `run!(bm)` / `step!`: explicit solver with one global time step (`stable_dt(bm)`; deep-water coarse cells and shallow-water
  fine cells usually have similar CFL limits because cg decreases with depth).
* GSE alleviation (`set_gse!(bm; ...)`) in all three (explicit sub-steps are synchronised over the blocks).
Interfaces between levels are first order and not exactly conservative (ghost values, no flux correction).

```julia
mesh = BlockMesh(; x0 = 0, y0 = 0, Lx = 20e3, Ly = 10e3, dx0 = 200.0, bs = 16, maxlevel = 3)   # 200 m … 25 m cells
refine_by_depth!(mesh, depth; resolution = d -> 5d)       # depth: (x, y) -> d (m, positive down) or (xb, yb, D) raster
bm   = BlockModel(mesh; depth = depth, spec = SpectralGrid(ns = 24, nth = 36), params = WaveParams(breaking = true))
set_boundaries!(bm; left = SpectrumBC(Hs = 2.0, Tp = 10.0, dir = 270.0))
solve_stationary!(bm)
b = bulk(bm)                                               # fields as vectors over the cells (see cell_centres)
H = to_grid(bm.mesh, b.Hs, xg, yg)                         # sample onto any regular grid for plotting
```
"""
module WaveBlocks

using Base.Threads
using Printf
using ..WaveAction
using ..WaveSweep
import ..WaveAction: NG, BoundaryCondition, apply_bc!, initialize!, set_boundaries!, set_wind!, step!, run!, bulk, total_action, set_gse!,
                     stable_dt, tendency!, advance!, sources!, update_environment!
import ..WaveSweep: solve_stationary!, run_implicit!, step_implicit!, hs_tm, sweep_cell!, update_dissipation!, gse_cross!,
                    make_workspaces

export BlockMesh, refine!, refine_by_depth!, balance!, nblocks, ncells, block_bounds, block_dx, cell_centres, cell_dx,
       cell_index, locate, grid_index, to_grid, bilinear, BlockModel, cell_depth, cell_wet, mesh_summary

# =====================================================================================================================
# Mesh
# =====================================================================================================================
"""
    BlockMesh(; x0 = 0.0, y0 = 0.0, Lx, Ly, dx0, bs = 16, maxlevel = 3)

Block quadtree over the rectangle `[x0, x0 + Lx] × [y0, y0 + Ly]`, initially `nbx × nby` level-0 blocks of `bs × bs` cells of
size `dx0` (the domain is extended to a whole number of blocks). Levels go up to `maxlevel` (finest cells `dx0 / 2^maxlevel`).
Refine with `refine_by_depth!` or `refine!`.
"""
mutable struct BlockMesh
    x0::Float64
    y0::Float64
    dx0::Float64
    bs::Int
    nbx::Int
    nby::Int
    maxlevel::Int
    level::Vector{Int}
    bx::Vector{Int}                       # block index at its own level (0-based)
    by::Vector{Int}
    index::Dict{NTuple{3,Int},Int}        # (level, bx, by) -> block number, leaves only
end

function BlockMesh(; x0 = 0.0, y0 = 0.0, Lx, Ly, dx0, bs::Int = 16, maxlevel::Int = 3)
    dx0 > 0 && Lx > 0 && Ly > 0 || error("Lx, Ly and dx0 must be positive")
    bs >= 2 * NG || error("bs must be at least $(2NG)")
    maxlevel >= 0 || error("maxlevel must be >= 0")
    nbx = max(1, ceil(Int, Lx / (bs * dx0) - 1e-9)); nby = max(1, ceil(Int, Ly / (bs * dx0) - 1e-9))
    lv = zeros(Int, nbx * nby); bx = [i for j in 0:nby-1 for i in 0:nbx-1]; by = [j for j in 0:nby-1 for i in 0:nbx-1]
    return BlockMesh(Float64(x0), Float64(y0), Float64(dx0), bs, nbx, nby, maxlevel, lv, bx, by)
end
function BlockMesh(x0, y0, dx0, bs, nbx, nby, maxlevel, level::Vector{Int}, bx::Vector{Int}, by::Vector{Int})
    m = BlockMesh(Float64(x0), Float64(y0), Float64(dx0), Int(bs), Int(nbx), Int(nby), Int(maxlevel), level, bx, by,
                  Dict{NTuple{3,Int},Int}())
    _sort!(m)
    _check_cover(m)
    return m
end

Base.show(io::IO, m::BlockMesh) = print(io, mesh_summary(m))

"""One-line description: blocks and cells per level, and the equivalent uniform finest grid."""
function mesh_summary(m::BlockMesh)
    s = @sprintf("BlockMesh: %d blocks of %d×%d cells, %d cells", nblocks(m), m.bs, m.bs, ncells(m))
    for l in 0:m.maxlevel
        n = count(==(l), m.level)
        n > 0 && (s *= @sprintf(";  level %d (dx = %g): %d blocks", l, block_dx(m, l), n))
    end
    nf = m.nbx * m.nby * m.bs^2 * 4^maximum(m.level)
    s *= @sprintf(";  %.1f%% of the uniform finest grid (%d cells)", 100 * ncells(m) / nf, nf)
    return s
end

nblocks(m::BlockMesh) = length(m.level)
ncells(m::BlockMesh) = nblocks(m) * m.bs^2
block_dx(m::BlockMesh, l::Integer) = m.dx0 / 2.0^l
_nb(m::BlockMesh, l) = (m.nbx << l, m.nby << l)
_inside(m::BlockMesh, l, bx, by) = (nx = m.nbx << l; ny = m.nby << l; 0 <= bx < nx && 0 <= by < ny)

"""(xlo, xhi, ylo, yhi) of block `k`."""
function block_bounds(m::BlockMesh, k::Integer)
    L = m.bs * block_dx(m, m.level[k])
    return (m.x0 + m.bx[k] * L, m.x0 + (m.bx[k] + 1) * L, m.y0 + m.by[k] * L, m.y0 + (m.by[k] + 1) * L)
end

# canonical order (south to north, west to east, by the block's lower-left corner) and lookup table
function _sort!(m::BlockMesh)
    key(k) = (m.by[k] << (m.maxlevel - m.level[k]), m.bx[k] << (m.maxlevel - m.level[k]), m.level[k])
    p = sortperm(1:nblocks(m); by = key)
    m.level = m.level[p]; m.bx = m.bx[p]; m.by = m.by[p]
    m.index = Dict((m.level[k], m.bx[k], m.by[k]) => k for k in 1:nblocks(m))
    length(m.index) == nblocks(m) || error("duplicate blocks in the mesh")
    return m
end

function _check_cover(m::BlockMesh)
    area = sum(4.0^(-l) for l in m.level; init = 0.0)
    isapprox(area, m.nbx * m.nby; rtol = 1e-12) || error("the blocks do not tile the domain")
    for k in 1:nblocks(m)
        _inside(m, m.level[k], m.bx[k], m.by[k]) || error("block $k is outside the domain")
        0 <= m.level[k] <= m.maxlevel || error("block $k has a level outside 0:maxlevel")
    end
end

# leaf containing the level-l block position (bx, by): (k, level), searching the same level and the coarser ones; (0, -1) if
# that region is refined further
function _leaf_up(m::BlockMesh, l, bx, by)
    L = l
    while L >= 0
        k = get(m.index, (L, bx, by), 0)
        k > 0 && return (k, L)
        L -= 1; bx >>= 1; by >>= 1
    end
    return (0, -1)
end

const _SIDE_CHILDREN = Dict(:left => ((0, 0), (0, 1)), :right => ((1, 0), (1, 1)), :bottom => ((0, 0), (1, 0)), :top => ((0, 1), (1, 1)))

# leaves of the level-l region (bx, by) that touch its `side`
function _touching!(out, m::BlockMesh, l, bx, by, side::Symbol)
    k, _ = _leaf_up(m, l, bx, by)
    if k > 0
        k in out || push!(out, k)
        return out
    end
    l < m.maxlevel || error("mesh hole at level $l block ($bx, $by)")
    for (a, b) in _SIDE_CHILDREN[side]
        _touching!(out, m, l + 1, 2bx + a, 2by + b, side)
    end
    return out
end

const _OPP = (left = :right, right = :left, bottom = :top, top = :bottom)
const _DIRS = ((:left, -1, 0), (:right, 1, 0), (:bottom, 0, -1), (:top, 0, 1))

"""Face neighbours of block `k` on `side` (:left, :right, :bottom, :top); empty at the domain boundary."""
function face_neighbours(m::BlockMesh, k::Integer, side::Symbol)
    l, bx, by = m.level[k], m.bx[k], m.by[k]
    for (s, dx, dy) in _DIRS
        s === side || continue
        _inside(m, l, bx + dx, by + dy) || return Int[]
        return _touching!(Int[], m, l, bx + dx, by + dy, _OPP[side])
    end
    error("side must be :left, :right, :bottom or :top")
end

function _split!(m::BlockMesh, flags::AbstractVector{Bool})
    lv, bx, by = Int[], Int[], Int[]
    for k in 1:nblocks(m)
        if flags[k] && m.level[k] < m.maxlevel
            for b in 0:1, a in 0:1
                push!(lv, m.level[k] + 1); push!(bx, 2m.bx[k] + a); push!(by, 2m.by[k] + b)
            end
        else
            push!(lv, m.level[k]); push!(bx, m.bx[k]); push!(by, m.by[k])
        end
    end
    m.level, m.bx, m.by = lv, bx, by
    _sort!(m)
end

"""
    balance!(mesh)

Refine blocks until face neighbours differ by at most one level (called by `refine!`).
"""
function balance!(m::BlockMesh)
    while true
        flags = falses(nblocks(m))
        for k in 1:nblocks(m), (side, _, _) in _DIRS
            nb = face_neighbours(m, k, side)
            any(n -> m.level[n] > m.level[k] + 1, nb) && (flags[k] = true)
        end
        any(flags) || return m
        _split!(m, flags)
    end
end

"""
    refine!(mesh, flags)   or   refine!(mesh) do k ... end

Split the flagged blocks (a Bool per block, or a function of the block number) into their four children, then restore
the 2:1 balance. Blocks at `maxlevel` are left as they are.
"""
function refine!(m::BlockMesh, flags::AbstractVector{Bool})
    length(flags) == nblocks(m) || error("one flag per block")
    _split!(m, flags)
    balance!(m)
end
refine!(f::Function, m::BlockMesh) = refine!(m, [Bool(f(k)) for k in 1:nblocks(m)])

"""
    bilinear(xb, yb, Z) -> (x, y) -> value

Bilinear interpolation in a raster `Z[i, j]` at coordinates `xb[i]`, `yb[j]` (ascending, uniform or not); values outside the
raster are taken from the nearest edge.
"""
function bilinear(xb::AbstractVector, yb::AbstractVector, Z::AbstractMatrix)
    size(Z) == (length(xb), length(yb)) || error("Z must be (length(xb), length(yb))")
    xs, ys, Zs = collect(Float64, xb), collect(Float64, yb), Matrix{Float64}(Z)
    function f(x, y)
        i = clamp(searchsortedlast(xs, x), 1, max(1, length(xs) - 1))
        j = clamp(searchsortedlast(ys, y), 1, max(1, length(ys) - 1))
        length(xs) == 1 && length(ys) == 1 && return Zs[1, 1]
        tx = length(xs) == 1 ? 0.0 : clamp((x - xs[i]) / (xs[i+1] - xs[i]), 0.0, 1.0)
        ty = length(ys) == 1 ? 0.0 : clamp((y - ys[j]) / (ys[j+1] - ys[j]), 0.0, 1.0)
        i2 = min(i + 1, length(xs)); j2 = min(j + 1, length(ys))
        return (1 - tx) * (1 - ty) * Zs[i, j] + tx * (1 - ty) * Zs[i2, j] + (1 - tx) * ty * Zs[i, j2] + tx * ty * Zs[i2, j2]
    end
    return f
end
_field(f::Function) = f
_field(t::Tuple{<:AbstractVector,<:AbstractVector,<:AbstractMatrix}) = bilinear(t...)
_field(c::Real) = (x, y) -> Float64(c)

"""
    refine_by_depth!(mesh, depth; resolution = d -> 5d, hmin = 0.05, nsub = 2, maxpass = mesh.maxlevel)

Refine from the bathymetry: a block is split while its cell size exceeds `resolution(d)`, `d` being the shallowest wet depth
(> `hmin`) found at `nsub × nsub` points per cell of the block (so features narrower than a cell are seen). Blocks without wet
points are not refined. `depth` is a function `(x, y) -> depth` (m, positive down) or a raster tuple `(xb, yb, D)`.

`resolution(d)` is the largest cell size you accept at depth `d` (m). The default `5d` gives 100 m cells at 20 m depth and
10 m cells at 2 m; the mesh's `dx0` and `maxlevel` bound the result.
"""
function refine_by_depth!(m::BlockMesh, depth; resolution = d -> 5d, hmin = 0.05, nsub::Int = 2, maxpass::Int = m.maxlevel)
    f = _field(depth)
    for _ in 1:maxpass
        flags = falses(nblocks(m))
        for k in 1:nblocks(m)
            l = m.level[k]
            l < m.maxlevel || continue
            xlo, _, ylo, _ = block_bounds(m, k)
            h = block_dx(m, l) / nsub
            n = m.bs * nsub
            dmin = Inf
            for jj in 1:n, ii in 1:n
                d = f(xlo + (ii - 0.5) * h, ylo + (jj - 0.5) * h)
                d > hmin && (dmin = min(dmin, d))
            end
            isfinite(dmin) || continue
            flags[k] = block_dx(m, l) > resolution(dmin) * (1 + 1e-9)
        end
        any(flags) || break
        refine!(m, flags)
    end
    return m
end

# ---- cells ----------------------------------------------------------------------------------------------------------
"""Global cell number of interior cell (i, j) of block k (cells are numbered block by block, i fastest)."""
cell_index(m::BlockMesh, k::Integer, i::Integer, j::Integer) = (k - 1) * m.bs^2 + (j - 1) * m.bs + i

"""
    cell_centres(mesh) -> (x, y)

Cell-centre coordinates of all cells (vectors of length `ncells`, in the order of `cell_index`).
"""
function cell_centres(m::BlockMesh)
    x = zeros(ncells(m)); y = zeros(ncells(m))
    for k in 1:nblocks(m)
        xlo, _, ylo, _ = block_bounds(m, k); h = block_dx(m, m.level[k])
        for j in 1:m.bs, i in 1:m.bs
            c = cell_index(m, k, i, j)
            x[c] = xlo + (i - 0.5) * h; y[c] = ylo + (j - 0.5) * h
        end
    end
    return x, y
end

"""Cell size of every cell (vector of length `ncells`)."""
cell_dx(m::BlockMesh) = vcat([fill(block_dx(m, l), m.bs^2) for l in m.level]...)

"""
    locate(mesh, x, y) -> (k, i, j) or nothing

Block and interior cell containing the point (x, y).
"""
function locate(m::BlockMesh, x::Real, y::Real)
    for l in 0:m.maxlevel
        L = m.bs * block_dx(m, l)
        bx = floor(Int, (x - m.x0) / L); by = floor(Int, (y - m.y0) / L)
        _inside(m, l, bx, by) || return nothing
        k = get(m.index, (l, bx, by), 0)
        if k > 0
            h = block_dx(m, l)
            i = clamp(floor(Int, (x - m.x0 - bx * L) / h) + 1, 1, m.bs)
            j = clamp(floor(Int, (y - m.y0 - by * L) / h) + 1, 1, m.bs)
            return (k, i, j)
        end
    end
    return nothing
end

"""
    grid_index(mesh, xg, yg) -> Matrix{Int}

Cell number containing each point of the regular grid `xg × yg` (0 outside the mesh); reuse it with `to_grid`.
"""
function grid_index(m::BlockMesh, xg::AbstractVector, yg::AbstractVector)
    I = zeros(Int, length(xg), length(yg))
    for j in eachindex(yg), i in eachindex(xg)
        r = locate(m, xg[i], yg[j])
        r === nothing || (I[i, j] = cell_index(m, r...))
    end
    return I
end

"""
    to_grid(mesh, v, xg, yg; outside = NaN)   or   to_grid(v, idx::Matrix{Int}; outside = NaN)

Sample a cell field `v` (vector over the cells, or array whose first dimension runs over the cells, e.g. (ncells, nt) or
(ncells, 1, nt)) onto the regular grid `xg × yg` (value of the cell containing each point): result (nxg, nyg, rest...).
"""
to_grid(m::BlockMesh, v::AbstractArray, xg::AbstractVector, yg::AbstractVector; outside = NaN) =
    to_grid(v, grid_index(m, xg, yg); outside = outside)
function to_grid(v::AbstractArray, idx::AbstractMatrix{<:Integer}; outside = NaN)
    rest = size(v)[2:end]
    V = reshape(v, size(v, 1), :)
    T = promote_type(eltype(v), typeof(outside))
    out = fill(convert(T, outside), size(idx)..., size(V, 2))
    for t in axes(V, 2), j in axes(idx, 2), i in axes(idx, 1)
        c = idx[i, j]
        c > 0 && (out[i, j, t] = V[c, t])
    end
    return reshape(out, size(idx)..., filter(!=(1), collect(rest))...)
end

# =====================================================================================================================
# Ghost-cell maps
# =====================================================================================================================
struct GhostMap
    dst::Vector{Int}          # linear index in the padded (nxg × nyg) plane of the block
    face::Vector{Bool}        # the ghost lies across a face (not in a corner region)
    ptr::Vector{Int}          # sources of dst[q]: ptr[q] : ptr[q+1]-1
    src_blk::Vector{Int}
    src_lin::Vector{Int}
    w::Vector{Float64}
end

_lin(m::BlockMesh, li, lj) = (NG + li + 1) + (NG + lj) * (m.bs + 2NG)      # 0-based interior indices -> padded linear index

# sources of the level-l global cell (gi, gj): one cell (same level / coarser leaf: injection) or the average of finer ones
function _sources!(out, m::BlockMesh, l, gi, gj, w)
    bs = m.bs
    k, L = _leaf_up(m, l, fld(gi, bs), fld(gj, bs))
    if k > 0
        ci, cj = fld(gi, 1 << (l - L)), fld(gj, 1 << (l - L))
        push!(out, (k, _lin(m, ci - m.bx[k] * bs, cj - m.by[k] * bs), w))
        return out
    end
    for b in 0:1, a in 0:1
        _sources!(out, m, l + 1, 2gi + a, 2gj + b, w / 4)
    end
    return out
end

function _ghostmap(m::BlockMesh, k::Integer)
    bs, l = m.bs, m.level[k]
    nxg = bs + 2NG
    ngx, ngy = m.nbx * bs << l, m.nby * bs << l
    dst = Int[]; face = Bool[]; ptr = [1]; sb = Int[]; sl = Int[]; w = Float64[]
    tmp = Tuple{Int,Int,Float64}[]
    for j in 1:nxg, i in 1:nxg
        ini = NG < i <= NG + bs; inj = NG < j <= NG + bs
        ini && inj && continue
        gi = m.bx[k] * bs + (i - NG - 1); gj = m.by[k] * bs + (j - NG - 1)
        (0 <= gi < ngx && 0 <= gj < ngy) || continue       # outside the domain: boundary condition
        empty!(tmp)
        _sources!(tmp, m, l, gi, gj, 1.0)
        push!(dst, i + (j - 1) * nxg); push!(face, ini || inj)
        for (b, s, ww) in tmp
            push!(sb, b); push!(sl, s); push!(w, ww)
        end
        push!(ptr, length(sb) + 1)
    end
    return GhostMap(dst, face, ptr, sb, sl, w)
end

# fill the ghosts of D (L × nplane) from the source matrices S[k] (L × nplane)
function _fill!(D::AbstractMatrix{Float64}, S::Vector{<:AbstractMatrix{Float64}}, gm::GhostMap, faces_only::Bool)
    L = size(D, 1)
    @inbounds for q in eachindex(gm.dst)
        faces_only && !gm.face[q] && continue
        d = gm.dst[q]
        r1, r2 = gm.ptr[q], gm.ptr[q+1] - 1
        if r1 == r2 && gm.w[r1] == 1.0
            Sk = S[gm.src_blk[r1]]; s = gm.src_lin[r1]
            for c in 1:L; D[c, d] = Sk[c, s]; end
        else
            for c in 1:L; D[c, d] = 0.0; end
            for r in r1:r2
                Sk = S[gm.src_blk[r]]; s = gm.src_lin[r]; ww = gm.w[r]
                for c in 1:L; D[c, d] += ww * Sk[c, s]; end
            end
        end
    end
    return D
end

# =====================================================================================================================
# Model
# =====================================================================================================================
const QUADS = ((1, 1), (-1, 1), (-1, -1), (1, -1))

"""
    BlockModel(mesh; depth, U = nothing, V = nothing, spec = SpectralGrid(), params = WaveParams(), depth_nsub = 1)

Wave model on a block mesh. `depth`, `U`, `V` are functions `(x, y) -> value`, raster tuples `(xb, yb, Z)` (bilinear) or
constants; they are sampled at the cell centres (`depth_nsub > 1`: averaged over `depth_nsub²` points per cell).
Boundary conditions (`set_boundaries!`) apply to the sides of the whole domain. Fields: `mesh`, `spec`, `params`,
`blocks` (one `WaveModel` per block), `bc`, `t`, `gse`.
"""
mutable struct BlockModel
    mesh::BlockMesh
    spec::SpectralGrid
    params::WaveParams
    blocks::Vector{WaveModel}
    maps::Vector{GhostMap}
    sides::Vector{NTuple{4,Bool}}                    # block side on the domain boundary (left, right, bottom, top)
    fronts::Vector{Vector{Vector{Int}}}              # per sweep quadrant: groups of independent blocks, in upwind order
    Nm::Vector{Matrix{Float64}}                      # (ns·nth × nplane) views of the blocks' N and N1
    N1m::Vector{Matrix{Float64}}
    bc::NTuple{4,BoundaryCondition}
    t::Float64
    gse::Union{Nothing,GSEParams}
end

function Base.setproperty!(bm::BlockModel, s::Symbol, v)
    if s === :params                                   # physics switches are read by every block
        for b in getfield(bm, :blocks); b.params = v; end
    elseif s === :t
        for b in getfield(bm, :blocks); b.t = v; end
    end
    setfield!(bm, s, convert(fieldtype(BlockModel, s), v))
end

Base.show(io::IO, bm::BlockModel) = print(io, "BlockModel on ", mesh_summary(bm.mesh), "; spectrum ", bm.spec.ns, "×", bm.spec.nth)

function _sample(f, xlo, ylo, h, bs, nsub)
    A = zeros(bs, bs)
    for j in 1:bs, i in 1:bs
        s = 0.0
        for b in 1:nsub, a in 1:nsub
            s += f(xlo + (i - 1 + (a - 0.5) / nsub) * h, ylo + (j - 1 + (b - 0.5) / nsub) * h)
        end
        A[i, j] = s / nsub^2
    end
    return A
end

function BlockModel(mesh::BlockMesh; depth, U = nothing, V = nothing, spec::SpectralGrid = SpectralGrid(),
                    params::WaveParams = WaveParams(), depth_nsub::Int = 1)
    fd = _field(depth); fu = U === nothing ? nothing : _field(U); fv = V === nothing ? nothing : _field(V)
    bs = mesh.bs
    nb = nblocks(mesh)
    blocks = Vector{WaveModel}(undef, nb)
    for k in 1:nb
        xlo, _, ylo, _ = block_bounds(mesh, k); h = block_dx(mesh, mesh.level[k])
        xc = [xlo + (i - 0.5) * h for i in 1:bs]; yc = [ylo + (j - 0.5) * h for j in 1:bs]
        D = max.(_sample(fd, xlo, ylo, h, bs, depth_nsub), 0.0)
        Uk = fu === nothing ? nothing : _sample(fu, xlo, ylo, h, bs, 1)
        Vk = fv === nothing ? nothing : _sample(fv, xlo, ylo, h, bs, 1)
        blocks[k] = WaveModel(CartesianGrid(xc, yc); depth = D, U = Uk, V = Vk, spec = spec, params = params)
    end
    maps = [_ghostmap(mesh, k) for k in 1:nb]
    sides = [(mesh.bx[k] == 0, mesh.bx[k] == (mesh.nbx << mesh.level[k]) - 1,
              mesh.by[k] == 0, mesh.by[k] == (mesh.nby << mesh.level[k]) - 1) for k in 1:nb]
    nplane = (bs + 2NG)^2
    Nm = [reshape(b.N, :, nplane) for b in blocks]; N1m = [reshape(b.N1, :, nplane) for b in blocks]
    bm = BlockModel(mesh, spec, params, blocks, maps, sides, _fronts(mesh), Nm, N1m,
                    (OpenBC(), OpenBC(), OpenBC(), OpenBC()), 0.0, nothing)
    _environment!(bm)
    set_boundaries!(bm)
    return bm
end

# environment ghosts from the neighbours, then the derived fields of every block
function _environment!(bm::BlockModel)
    nplane = (bm.mesh.bs + 2NG)^2
    for f in (:d, :U, :V, :dddt)
        S = [reshape(getfield(b.env, f), 1, nplane) for b in bm.blocks]
        for k in eachindex(bm.blocks)
            _fill!(S[k], S, bm.maps[k], false)
        end
    end
    for b in bm.blocks
        update_environment!(b; fill_ghosts = false)
    end
    return bm
end

# blocks grouped into wavefronts for each sweep quadrant: a block depends on its face neighbours on the upwind sides
function _fronts(m::BlockMesh)
    nb = nblocks(m)
    cx = zeros(nb); cy = zeros(nb)
    for k in 1:nb
        a, b, c, d = block_bounds(m, k); cx[k] = (a + b) / 2; cy[k] = (c + d) / 2
    end
    nbr = [Dict(s => face_neighbours(m, k, s) for (s, _, _) in _DIRS) for k in 1:nb]
    out = Vector{Vector{Vector{Int}}}()
    for (sx, sy) in QUADS
        order = sortperm(sx .* cx .+ sy .* cy)
        lev = zeros(Int, nb)
        for k in order
            lv = 1
            for n in nbr[k][sx > 0 ? :left : :right]
                lev[n] > 0 || error("block ordering error")
                lv = max(lv, lev[n] + 1)
            end
            for n in nbr[k][sy > 0 ? :bottom : :top]
                lev[n] > 0 || error("block ordering error")
                lv = max(lv, lev[n] + 1)
            end
            lev[k] = lv
        end
        groups = [Int[] for _ in 1:maximum(lev)]
        for k in order; push!(groups[lev[k]], k); end
        push!(out, groups)
    end
    return out
end

"""Depth of every cell (vector over the cells)."""
cell_depth(bm::BlockModel) = vcat([vec(b.env.d[NG+1:NG+bm.mesh.bs, NG+1:NG+bm.mesh.bs]) for b in bm.blocks]...)
"""Wet flag of every cell."""
cell_wet(bm::BlockModel) = vcat([vec(b.env.wet[NG+1:NG+bm.mesh.bs, NG+1:NG+bm.mesh.bs]) for b in bm.blocks]...)
cell_centres(bm::BlockModel) = cell_centres(bm.mesh)
ncells(bm::BlockModel) = ncells(bm.mesh)
nblocks(bm::BlockModel) = nblocks(bm.mesh)

# ---- ghost exchange ---------------------------------------------------------------------------------------------------
function _exchange!(bm::BlockModel, which::Symbol, t)
    M = which === :N ? bm.Nm : bm.N1m
    @threads :static for k in eachindex(bm.blocks)
        _fill!(M[k], M, bm.maps[k], false)
        b = bm.blocks[k]
        apply_bc!(which === :N ? b.N : b.N1, b, t)
    end
    return bm
end

# ---- set-up API ---------------------------------------------------------------------------------------------------------
"""
    set_boundaries!(bm; left, right, bottom, top)

Boundary conditions of the domain sides (`OpenBC()`, `ZeroBC()`, `SpectrumBC(...)`), as for a `WaveModel`.
"""
function set_boundaries!(bm::BlockModel; left = bm.bc[1], right = bm.bc[2], bottom = bm.bc[3], top = bm.bc[4])
    bm.bc = (left, right, bottom, top)
    for (k, b) in enumerate(bm.blocks)
        s = bm.sides[k]
        set_boundaries!(b; left = s[1] ? left : ExternalBC(), right = s[2] ? right : ExternalBC(),
                        bottom = s[3] ? bottom : ExternalBC(), top = s[4] ? top : ExternalBC())
    end
    _exchange!(bm, :N, bm.t)
    return bm
end

"""
    initialize!(bm; Hs = 0, Tp = 10, dir = 270, gamma = 3.3, spread = 10)

Uniform initial spectrum on all wet cells (calm by default); resets the time to 0.
"""
function initialize!(bm::BlockModel; kw...)
    for b in bm.blocks
        initialize!(b; kw...)
    end
    bm.t = 0.0
    _exchange!(bm, :N, 0.0)
    return bm
end

"""
    set_wind!(bm; U10, dir)   or   set_wind!(bm; u, v)

Uniform wind (nautical from, degrees), or components `u`, `v` (m/s) as functions `(x, y) ->` value / rasters `(xb, yb, Z)`.
"""
function set_wind!(bm::BlockModel; U10 = nothing, dir = nothing, u = nothing, v = nothing)
    if u !== nothing && v !== nothing
        fu, fv = _field(u), _field(v)
        for (k, b) in enumerate(bm.blocks)
            xlo, _, ylo, _ = block_bounds(bm.mesh, k); h = block_dx(bm.mesh, bm.mesh.level[k])
            set_wind!(b; u = _sample(fu, xlo, ylo, h, bm.mesh.bs, 1), v = _sample(fv, xlo, ylo, h, bm.mesh.bs, 1))
        end
    else
        for b in bm.blocks
            set_wind!(b; U10 = U10, dir = dir)
        end
    end
    return bm
end

"""
    set_gse!(bm; alpha_s = 3.0, alpha_n = 3.0, cfl = 1.0, tau = 0.0, maxsub = 400)   /   set_gse!(bm, nothing)

Garden-sprinkler-effect alleviation on every block (see `GSEParams`). With `cfl > 0` the filter scales with each block's own
cell size.
"""
function set_gse!(bm::BlockModel; kw...)
    for b in bm.blocks
        set_gse!(b; kw...)
    end
    bm.gse = bm.blocks[1].gse
    return bm
end
function set_gse!(bm::BlockModel, ::Nothing)
    for b in bm.blocks
        set_gse!(b, nothing)
    end
    bm.gse = nothing
    return bm
end

# =====================================================================================================================
# Solvers
# =====================================================================================================================
function _sweep_block!(b::WaveModel, ws, sx, sy, dtinv, sw, gsel)
    n = b.grid.nx
    wet = b.env.wet
    for v in 1:n, u in 1:n
        i = (sx > 0 ? u : n + 1 - u) + NG
        j = (sy > 0 ? v : n + 1 - v) + NG
        wet[i, j] && sweep_cell!(b, ws, i, j, sx, sy, dtinv, b.N1, sw, gsel)
    end
    return b
end

# one iteration of the four sweeps over the whole mesh (Gauss–Seidel across the blocks, in upwind order)
function _four_sweeps!(bm::BlockModel, dtinv, sw::SweepParams, wss, tbc; gsel = nothing)
    for (q, (sx, sy)) in enumerate(QUADS)
        _exchange!(bm, :N, tbc)
        for b in bm.blocks
            update_dissipation!(b)
        end
        if gsel !== nothing
            for b in bm.blocks
                gse_cross!(b.dN, b.N, b, gsel)
            end
        end
        for group in bm.fronts[q]
            @threads :static for k in group
                b = bm.blocks[k]
                _fill!(bm.Nm[k], bm.Nm, bm.maps[k], true)          # face ghosts from the latest neighbour values
                apply_bc!(b.N, b, tbc)
                _sweep_block!(b, wss[threadid()], sx, sy, dtinv, sw, gsel)
            end
        end
    end
    _exchange!(bm, :N, tbc)
    return bm
end

"""
    hs_tm(bm) -> (Hs, Tm01)     vectors over the cells
"""
function hs_tm(bm::BlockModel)
    n2 = bm.mesh.bs^2
    Hs = zeros(ncells(bm)); Tm = zeros(ncells(bm))
    for (k, b) in enumerate(bm.blocks)
        H, T = hs_tm(b)
        r = (k - 1) * n2 + 1:k * n2
        Hs[r] .= vec(H); Tm[r] .= vec(T)
    end
    return Hs, Tm
end

"""
    solve_stationary!(bm; sweep = SweepParams(), verbose = true) -> (iterations, converged)

Stationary solution on the block mesh (same convergence test as for a `WaveModel`, over all wet cells).
"""
function solve_stationary!(bm::BlockModel; sweep::SweepParams = SweepParams(), verbose::Bool = true)
    wss = make_workspaces(bm.blocks[1])
    wet = cell_wet(bm)
    nwet = count(wet)
    gsel = bm.gse
    if gsel !== nothing && gsel.tau <= 0 && gsel.cfl <= 0
        verbose && println("  note: stationary mode has no time step; GSE filter uses cfl = 1 (Ts = Δx/cg)")
        gsel = GSEParams(alpha_s = gsel.alpha_s, alpha_n = gsel.alpha_n, cfl = 1.0, tau = 0.0, maxsub = gsel.maxsub)
    end
    nwet == 0 && return (0, true)
    Hso, Tmo = hs_tm(bm)
    for iter in 1:sweep.maxiter
        _four_sweeps!(bm, 0.0, sweep, wss, bm.t; gsel = gsel)
        Hs, Tm = hs_tm(bm)
        ok = 0
        for c in eachindex(Hs)
            wet[c] || continue
            okH = abs(Hs[c] - Hso[c]) <= max(sweep.dabs, sweep.drel * Hs[c])
            okT = abs(Tm[c] - Tmo[c]) <= max(sweep.dabs, sweep.drel * Tm[c])
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

"""
    step_implicit!(bm, dt, sw = SweepParams(); wss)

One backward-Euler step of length `dt` on the block mesh.
"""
function step_implicit!(bm::BlockModel, dt, sw::SweepParams = SweepParams(); wss = make_workspaces(bm.blocks[1]))
    for b in bm.blocks
        copyto!(b.N1, b.N)
    end
    for _ in 1:sw.iters
        _four_sweeps!(bm, 1 / dt, sw, wss, bm.t + dt)
    end
    bm.t = bm.t + dt
    _gse_step!(bm, dt)
    return bm
end

"""Largest stable explicit time step over all blocks."""
stable_dt(bm::BlockModel, cfl = 0.4) = minimum(stable_dt(b, cfl) for b in bm.blocks)

"""
    step!(bm, dt)

One explicit predictor/corrector step on the block mesh (ghosts exchanged before each stage).
"""
function step!(bm::BlockModel, dt)
    t = bm.t
    _exchange!(bm, :N, t)
    for b in bm.blocks
        tendency!(b.dN, b.N, b); advance!(b.N1, b.N, b.dN, 0.5dt, b)
    end
    _exchange!(bm, :N1, t + 0.5dt)
    for b in bm.blocks
        tendency!(b.dN, b.N1, b); advance!(b.N, b.N, b.dN, dt, b)
    end
    for b in bm.blocks
        sources!(b, dt)
    end
    bm.t = t + dt
    _gse_step!(bm, dt)
    return bm
end

# GSE filter over one solver step: explicit sub-steps, the same number on every block, ghosts exchanged in between
function _gse_step!(bm::BlockModel, dt)
    bm.gse === nothing && return 0
    caches = [b.gsec === nothing ? WaveAction.gse_prepare!(b) : b.gsec for b in bm.blocks]
    nsub = 1
    for (b, c) in zip(bm.blocks, caches)
        sc = c.dtmode ? dt : 1.0
        nsub = max(nsub, ceil(Int, b.gse.safety * dt * c.Dsum * sc / min(b.grid.dx, b.grid.dy)^2))
    end
    if nsub > bm.gse.maxsub
        @warn "GSE filter needs $nsub explicit sub-steps (> maxsub = $(bm.gse.maxsub)) on the block mesh; capped (shorten dt)" maxlog = 1
        nsub = bm.gse.maxsub
    end
    dτ = dt / nsub
    for s in 1:nsub
        _exchange!(bm, :N, bm.t)
        for (b, c) in zip(bm.blocks, caches)
            WaveAction.gse_explicit_substep!(b, c, dτ, c.dtmode ? dt : 1.0, s == nsub)
        end
    end
    _exchange!(bm, :N, bm.t)
    return nsub
end

function _finite(bm::BlockModel)
    n = bm.mesh.bs
    all(b -> all(isfinite, @view b.N[:, :, NG+1:NG+n, NG+1:NG+n]), bm.blocks)
end

"""
    run!(bm; tend, cfl = 0.4, dt = nothing, output_dt = nothing, callback = nothing, verbose = true)

Explicit run to absolute time `tend` on the block mesh (global time step `stable_dt(bm, cfl)` unless `dt` is given).
"""
function run!(bm::BlockModel; tend, cfl = 0.4, dt = nothing, output_dt = nothing, callback = nothing, verbose = true)
    dtmax = dt === nothing ? stable_dt(bm, cfl) : dt
    isfinite(dtmax) || error("no wet cells: nothing to compute")
    verbose && println("WaveBlocks explicit: dt = $(round(dtmax, digits = 4)) s, threads = $(nthreads()), ", mesh_summary(bm.mesh))
    nextout = output_dt === nothing ? Inf : bm.t + output_dt
    callback === nothing || callback(bm, bm.t)
    nsteps = 0
    tol = 1e-9 * max(1.0, abs(tend))
    while bm.t < tend - tol
        h = min(dtmax, tend - bm.t, nextout - bm.t)
        step!(bm, h)
        nsteps += 1
        if bm.t >= nextout - tol
            callback === nothing || callback(bm, bm.t)
            nextout += output_dt
        end
        _finite(bm) || error("non-finite wave action at t = $(bm.t) s")
    end
    return nsteps
end

"""
    run_implicit!(bm; tend, dt = 10 stable_dt, sweep = SweepParams(), output_dt = nothing, callback = nothing, verbose = true)

Implicit non-stationary run on the block mesh.
"""
function run_implicit!(bm::BlockModel; tend, dt = nothing, sweep::SweepParams = SweepParams(), output_dt = nothing,
                       callback = nothing, verbose = true)
    dtmax = dt === nothing ? 10 * stable_dt(bm, 0.5) : dt
    isfinite(dtmax) || error("no wet cells: nothing to compute")
    wss = make_workspaces(bm.blocks[1])
    verbose && println("WaveBlocks implicit: dt = $(round(dtmax, digits = 3)) s, threads = $(nthreads()), ", mesh_summary(bm.mesh))
    nextout = output_dt === nothing ? Inf : bm.t + output_dt
    callback === nothing || callback(bm, bm.t)
    nsteps = 0
    tol = 1e-9 * max(1.0, abs(tend))
    while bm.t < tend - tol
        h = min(dtmax, tend - bm.t, nextout - bm.t)
        step_implicit!(bm, h, sweep; wss = wss)
        nsteps += 1
        if bm.t >= nextout - tol
            callback === nothing || callback(bm, bm.t)
            nextout += output_dt
        end
        _finite(bm) || error("non-finite wave action at t = $(bm.t) s")
    end
    return nsteps
end

# =====================================================================================================================
# Diagnostics
# =====================================================================================================================
"""
    bulk(bm) -> (Hs, Tm, Dir, Sxx, Sxy, Syy)

Vectors over the cells (order of `cell_index` / `cell_centres`); map them to a regular grid with `to_grid`.
"""
function bulk(bm::BlockModel)
    n2 = bm.mesh.bs^2; nc = ncells(bm)
    out = (Hs = zeros(nc), Tm = zeros(nc), Dir = zeros(nc), Sxx = zeros(nc), Sxy = zeros(nc), Syy = zeros(nc))
    for (k, b) in enumerate(bm.blocks)
        r = (k - 1) * n2 + 1:k * n2
        o = bulk(b)
        for f in keys(out)
            getfield(out, f)[r] .= vec(getfield(o, f))
        end
    end
    return out
end

"""Total wave action Σ N dσ dθ dA over all blocks."""
total_action(bm::BlockModel) = sum(total_action(b) for b in bm.blocks)

end # module
