# Block-adaptive mesh (`WaveBlocks`)

Coarse cells in deep water, fine cells where the bathymetry is shallow and complex. The mesh is a quadtree of square
blocks of `bs × bs` cells (16 by default, as in BG_Flood). Each refinement level halves the cell size, and face-neighbouring
blocks differ by at most one level.

```julia
using BinWaves
mesh = BlockMesh(; x0 = 0, y0 = 0, Lx = 20e3, Ly = 10e3, dx0 = 160.0, maxlevel = 3)   # 160, 80, 40, 20 m cells
refine_by_depth!(mesh, depth; resolution = d -> 5d)      # depth: (x, y) -> d, or a raster (xb, yb, D)
println(mesh)                                            # blocks / cells per level, % of the uniform finest grid
bm = BlockModel(mesh; depth = depth, spec = SpectralGrid(ns = 24, nth = 36), params = WaveParams(breaking = true))
set_boundaries!(bm; left = SpectrumBC(Hs = 2.0, Tp = 10.0, dir = 260.0))
solve_stationary!(bm)                                    # also run!(bm; tend) and run_implicit!(bm; tend, dt)
b = bulk(bm)                                             # Hs, Tm, Dir, Sxx, … as vectors over the cells
Hs = to_grid(bm.mesh, b.Hs, xg, yg)                      # sample onto any regular grid (plotting, output)
```
`set_boundaries!`, `initialize!`, `set_wind!`, `set_gse!`, `stable_dt`, `total_action`, `hs_tm` work as on a `WaveModel`.
Cell fields are vectors in `cell_index` order (block by block, i fastest): `cell_centres(mesh)`, `cell_dx(mesh)`,
`cell_depth(bm)`, `cell_wet(bm)`; `locate(mesh, x, y)` finds a cell, `grid_index` + `to_grid` resample (`interp = :bilinear`
for interpolation). For files, see `docs/README_Output.md` (CF NetCDF, native levels or the finest grid).

## Refinement
`refine_by_depth!(mesh, depth; resolution, nsub = 2)` splits a block while its cell size is larger than `resolution(d)`, `d`
being the shallowest wet depth found at `nsub × nsub` points per cell of the block. Coastlines and shoals therefore get the
finest cells and deep water keeps the coarse ones. `resolution(d)` is the largest cell size you accept at depth `d`. The
default `5d` gives 100 m at 20 m depth and 10 m at 2 m; `dx0` and `maxlevel` bound it. For any other criterion use
`refine!(mesh) do k ... end` (a Bool per block number; `block_bounds(mesh, k)` gives its extent). The 2:1 balance is
restored automatically.

## How it works
* Every block is an ordinary `WaveModel` on its own 16 × 16 grid with two ghost layers. All the cell kernels (transport,
  refraction, sources, implicit sweep update, GSE) are the uniform-grid ones, unchanged.
* The ghost cells (and the ghost depths and currents, so face group velocities and refraction gradients see the
  neighbours) are filled from the neighbouring blocks:
  * same level: copy;
  * fine side of an interface: the containing coarse cell (injection);
  * coarse side: the average of the four fine cells (restriction).
  The domain sides use the boundary conditions as usual. Inside the mesh the blocks' sides are marked `ExternalBC()`.
* Stationary and implicit solvers: each of the four sweeps visits the blocks in its upwind order, as a Gauss–Seidel
  sweep over blocks, with ghosts refreshed from the latest neighbour values. Blocks that do not depend on each other (no
  shared face across the sweep's upwind sides) form wavefronts and run in parallel on threads.
* Explicit solver: one global time step, `stable_dt(bm)` = minimum over the blocks. Fine cells sit in shallow water
  where the group velocity is small, so the step is usually close to that of the coarse deep-water cells. A fine block in
  deep water does set a small step, though.
* GSE alleviation: per block, with its own cell size in `Ts = cfl·Δx/cg`. The explicit sub-steps are synchronised across
  the blocks (ghosts exchanged between them). The ADI branch is not used on a block mesh.

## Accuracy and cost (test/test_blocks.jl and a larger case)
* **Uniform mesh** (all blocks at one level): bit-for-bit identical to the single-grid model for the explicit, implicit and
  stationary solvers, with and without GSE.
* **Uniform deep-water swell** crossing level 0/1/2 interfaces: Hs stays uniform to 1e-8 m.
* **Beach with a shoal**, 3.2 × 1.6 km, 50 / 25 / 12.5 m cells: Hs within 1.5 mm RMS (max 3 cm) of the uniform 12.5 m grid,
  with 53 % of the cells in half the time.
* **Shelf with a bay**, 20 × 10 km, 60 m deep offshore, 160 / 80 / 40 / 20 m cells (`resolution = d -> 5d`): 128,000 cells,
  24 % of the uniform 20 m grid. Stationary solve in 130 s against 436 s (4 threads). Hs within 3.1 mm RMS (max 3 cm) of
  the uniform 20 m grid, 3.4 mm RMS where the depth is below 5 m. The cost per cell per iteration is the same as on a
  uniform grid.

## BinWaves on a block mesh
`build_kp` and `build_wind_kp` take a `BlockModel`. Give the output `points` as `(x, y)` coordinates. The coefficient
fields are then `(ncase, ncells, 1)` arrays over the mesh cells, and the mesh is stored in `kp.info["mesh"]` and in the
NetCDF files (dimension `cell`, with `cell_x`, `cell_y` and the block list). `reconstruct` returns `Hs` as `(ncells, 1, nt)`;
`to_grid(kp, rec.Hs, xg, yg)` samples it onto a regular grid. `rec.Dir` and `rec.Spread` (mean direction and directional
spread) have the same shape. Superposition is as exact as on a regular grid
(3.5e-7 relative error in the test).

## Limits
* Level interfaces are first order (injection / restriction, the MUSCL slope is lost at the interface) and not exactly
  conservative (no flux correction). The errors measured above are a few mm in Hs.
* Depth, currents and wind are sampled at the cell centres (`depth_nsub` averages the depth over sub-points). There is no
  `set_environment!` on a block mesh yet: rebuild the `BlockModel` for a new depth.
* The mesh is static (built once from the bathymetry, not adapted to the wave field during a run).
* The GPU backend (`WaveGPU`) does not support block meshes yet.
