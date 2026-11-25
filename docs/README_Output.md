# CF NetCDF output of bulk fields

`BulkWriter` writes bulk wave fields to CF-1.8 NetCDF files, one record per output time (unlimited `time` dimension). It
works on a block mesh (`BlockModel`, `BlockMesh`, or BinWaves coefficients built on one) and on a regular grid (`WaveModel`).
Files written by the tests pass the CF checker (`cfchecks -v 1.8`) with 0 errors and 0 warnings in every layout.

```julia
# one snapshot
save_bulk_netcdf("waves.nc", bm; layout = :both, vars = (:Hs, :Tm, :Dir))

# a time series from a run (record every 10 min)
w = BulkWriter("run.nc", bm; layout = :levels, vars = (:Hs, :Tm, :Dir, :Sxx, :Sxy, :Syy),
               time_units = "seconds since 2026-01-01 00:00:00", crs = Dict("grid_mapping_name" => "transverse_mercator", ...))
run!(bm; tend = 6 * 3600, output_dt = 600, callback = (m, t) -> write_bulk!(w, m))
close(w)

# BinWaves reconstruction (any cell fields, last dimension = time)
rec = reconstruct(kp, E, f, dir)
save_fields_netcdf("rec.nc", kp, times; Hs = rec.Hs, Dir = rec.Dir, Spread = rec.Spread, layout = :finest)
```

## Layouts on a block mesh
* **`:levels`** stores the native mesh. There is one regular grid per refinement level present, covering the whole domain
  at that level's cell size, with dimensions `x_L0, y_L0`, `x_L1, y_L1`, …. Each field becomes one variable per level
  (`Hs_L0(time, y_L0, x_L0)`, `Hs_L1`, …). A cell's value sits in the layer of its own level and is NaN (`_FillValue`) in
  the others, so every wet cell appears in exactly one layer. Overlaying the layers from coarse to fine gives the mesh
  solution with nothing interpolated. Each variable carries `refinement_level` and `cell_size` attributes, and there is a
  `depth_L<l>` per level.
* **`:finest`** puts every field on the grid of the finest level present, as a single layer (`Hs(time, y, x)`).
  * `interp = :bilinear` (default) interpolates between the cell centres of the level the point belongs to. A centre
    inside a finer region takes the average of its fine cells; a centre in a coarser block takes that coarse cell. Dry
    cells and centres outside the domain are left out, and the remaining weights are renormalised.
  * The finest cells keep their own values exactly. A linear field is reproduced exactly wherever the four surrounding
    centres are at the same level and wet. Elsewhere the error is within half a coarse cell of the local gradient.
  * Directions are interpolated through their unit vector, so there is no averaging across north.
  * `interp = :nearest` copies the value of the containing cell instead (blocky, but no smoothing).
  * A `level` variable records which refinement level each point comes from, and `depth` is included.
* **`:both`** writes both layouts in one file.

A regular grid always gives a single `(time, y, x)` layer.

## Metadata
* **Fields:** `Hs` → `sea_surface_wave_significant_height` (m); `Tm`/`Tm01` →
  `sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment` (s); `Tp` →
  `sea_surface_wave_period_at_variance_spectral_density_maximum` (s); `Dir` → `sea_surface_wave_from_direction` (degree,
  nautical "from"); `Spread` → `sea_surface_wave_directional_spread` (degree). `Sxx/Sxy/Syy` get a long name and units N m-1 (CF has no standard name for them). `depth` →
  `sea_floor_depth_below_sea_surface` (m). Other names get only a long name; add attributes with
  `attributes = Dict(:name => Dict("standard_name" => ..., "units" => ...))`.
* **Coordinates:** `x`, `y` are `projection_x/y_coordinate` in metres. Pass `crs` (a Dict of CF grid-mapping attributes:
  `grid_mapping_name` and its parameters, and optionally `crs_wkt` with a real WKT string) to add a `crs` variable that
  every field references through `grid_mapping`.
* **Time:** `standard_name = time`, with your `time_units`, calendar `standard`.
* **Missing values:** dry cells are NaN. `write_bulk!` also sets `Tm` and `Dir` to NaN where Hs < 1 mm.
* **Mesh:** the global attributes `mesh_x0, mesh_y0, mesh_dx0, mesh_bs, mesh_nbx, mesh_nby, mesh_maxlevel` and the
  variables `mesh_level, mesh_bx, mesh_by` (dimension `block`) describe the mesh, so it can be rebuilt with
  `BlockMesh(x0, y0, dx0, bs, nbx, nby, maxlevel, level, bx, by)`.
* **Storage:** Float32 by default (`T = Float64` for double), deflate level 4 (`compress`). The file is synced after every
  record.

The same interpolation is available in memory: `to_grid(mesh, v, xg, yg; interp = :bilinear)`, or build the weights once
with `interp_stencil(mesh, xg, yg)` and apply them with `to_grid(v, stencil)`.
