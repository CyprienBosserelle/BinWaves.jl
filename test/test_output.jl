# Tests of the CF NetCDF output of bulk fields (native levels / finest grid / regular grid).
using BinWaves, NetCDF
using Test

@testset "CF NetCDF output" begin
    depth(x, y) = x < 2900 ? 60.0 * (1 - x / 3000) + 1.0 : 0.0
    mesh = BlockMesh(; Lx = 3200.0, Ly = 1600.0, dx0 = 100.0, maxlevel = 2)
    refine_by_depth!(mesh, depth; resolution = d -> 8d)
    @test length(unique(mesh.level)) == 3
    bm = BlockModel(mesh; depth = depth, spec = SpectralGrid(ns = 6, nth = 12))
    x, y = cell_centres(mesh)
    wet = cell_wet(bm)
    lin = 1.0 .+ 2e-4 .* x .+ 5e-4 .* y                       # a linear field: bilinear interpolation reproduces it
    dir = ifelse.(x .< 1600, 355.0, 5.0)                     # straddles north: must interpolate through 0, not 180
    dirn = mktempdir()

    @testset "native levels" begin
        f = joinpath(dirn, "levels.nc")
        w = BulkWriter(f, bm; layout = :levels, vars = (:Hs, :Dir))
        write_fields!(w, 0.0; Hs = lin, Dir = dir)
        write_fields!(w, 600.0; Hs = 2 .* lin, Dir = dir)
        close(w)
        @test ncread(f, "time") == [0.0, 600.0]
        @test ncgetatt(f, "global", "Conventions") == "CF-1.8"
        @test ncgetatt(f, "Hs_L0", "standard_name") == "sea_surface_wave_significant_height"
        @test ncgetatt(f, "Dir_L2", "standard_name") == "sea_surface_wave_from_direction"
        nfound = 0
        for l in sort(unique(mesh.level))
            H = ncread(f, "Hs_L$l"); D = ncread(f, "depth_L$l")
            lx, ly = level_grid(mesh, l)
            @test size(H) == (length(lx), length(ly), 2)
            @test ncread(f, "x_L$l") ≈ lx
            for c in 1:ncells(mesh)
                k = (c - 1) ÷ mesh.bs^2 + 1
                i = round(Int, (x[c] - mesh.x0) / block_dx(mesh, l) + 0.5); j = round(Int, (y[c] - mesh.y0) / block_dx(mesh, l) + 0.5)
                if mesh.level[k] == l
                    if wet[c]
                        @test H[i, j, 1] ≈ lin[c] rtol = 1e-6
                        @test H[i, j, 2] ≈ 2lin[c] rtol = 1e-6
                        nfound += 1
                    else
                        @test isnan(H[i, j, 1]) && isnan(D[i, j])
                    end
                end
            end
            # where this level has no cells, the layer is NaN
            @test count(!isnan, H[:, :, 1]) == count(c -> wet[c] && mesh.level[(c - 1) ÷ mesh.bs^2 + 1] == l, 1:ncells(mesh))
        end
        @test nfound == count(wet)                   # every wet cell in exactly one layer
        @test ncread(f, "mesh_level") == mesh.level
    end

    @testset "interpolated to the finest level" begin
        f = joinpath(dirn, "finest.nc")
        save_fields_netcdf(f, bm, [0.0, 3600.0]; layout = :finest, Hs = hcat(lin, lin), Dir = hcat(dir, dir))
        fx, fy = finest_grid(mesh)
        H = ncread(f, "Hs")[:, :, 1]; D = ncread(f, "Dir")[:, :, 1]; L = ncread(f, "level")
        @test size(H) == (length(fx), length(fy))
        exact = [1.0 + 2e-4 * a + 5e-4 * b for a in fx, b in fy]
        lf = maximum(mesh.level)
        ok = .!isnan.(H)
        @test all(abs.(H[ok .& (L .== lf)] .- exact[ok .& (L .== lf)]) .< 1e-5)     # finest cells: their own values
        # elsewhere within half a coarse cell of gradient (exact where all four centres are same-level and wet, see below)
        @test maximum(abs.(H[ok] .- exact[ok])) < sqrt(2e-4^2 + 5e-4^2) * mesh.dx0 / 2
        # on a single-level mesh a linear field is reproduced exactly away from the domain edge
        mu = BlockMesh(; Lx = 3200.0, Ly = 1600.0, dx0 = 50.0, maxlevel = 0)
        xu, yu = cell_centres(mu)
        st = interp_stencil(mu, collect(30.0:20.0:3170.0), collect(30.0:20.0:1570.0))
        Gu = to_grid(1.0 .+ 2e-4 .* xu .+ 5e-4 .* yu, st)
        @test maximum(abs.(Gu .- [1.0 + 2e-4 * a + 5e-4 * b for a in 30.0:20.0:3170.0, b in 30.0:20.0:1570.0])) < 1e-12
        @test all(isnan, H[fx .> 2950, :])                                           # dry
        @test all(d -> min(d, 360 - d) <= 5 + 1e-3, D[.!isnan.(D)])                   # circular interpolation
        @test ncgetatt(f, "Hs", "comment")[1:12] == "interpolated"
        # nearest
        g = joinpath(dirn, "nearest.nc")
        save_fields_netcdf(g, bm, [0.0]; layout = :finest, interp = :nearest, Hs = lin)
        Hn = ncread(g, "Hs")[:, :, 1]
        @test Hn ≈ to_grid(mesh, ifelse.(wet, lin, NaN), fx, fy) nans = true
    end

    @testset "both layouts, model output, regular grid" begin
        set_boundaries!(bm; left = SpectrumBC(Hs = 1.0, Tp = 8.0, dir = 270.0))
        solve_stationary!(bm; sweep = SweepParams(maxiter = 3), verbose = false)
        f = joinpath(dirn, "both.nc")
        save_bulk_netcdf(f, bm; layout = :both, vars = (:Hs, :Tm, :Dir, :Sxx, :Sxy, :Syy), t = 7200.0,
                         crs = Dict("grid_mapping_name" => "transverse_mercator", "longitude_of_central_meridian" => 173.0,
                                    "latitude_of_projection_origin" => 0.0, "scale_factor_at_central_meridian" => 0.9996,
                                    "false_easting" => 1.6e6, "false_northing" => 1e7))
        b = bulk(bm)
        @test ncgetatt(f, "Hs", "grid_mapping") == "crs" && ncgetatt(f, "crs", "grid_mapping_name") == "transverse_mercator"
        Hf = ncread(f, "Hs")[:, :, 1]
        lf = maximum(mesh.level); L = ncread(f, "level")
        fx, fy = finest_grid(mesh)
        Hc = to_grid(mesh, b.Hs, fx, fy)
        sel = (L .== lf) .& .!isnan.(Hf)
        @test any(sel) && Hf[sel] ≈ Float32.(Hc[sel])
        @test ncread(f, "time") == [7200.0]
        # regular grid
        xc = collect(25.0:50.0:3175.0); yc = collect(25.0:50.0:1575.0)
        m = WaveModel(CartesianGrid(xc, yc); depth = [depth(a, b) for a in xc, b in yc], spec = SpectralGrid(ns = 6, nth = 12))
        set_boundaries!(m; left = SpectrumBC(Hs = 1.0, Tp = 8.0, dir = 270.0))
        solve_stationary!(m; sweep = SweepParams(maxiter = 3), verbose = false)
        g = joinpath(dirn, "grid.nc")
        save_bulk_netcdf(g, m; vars = (:Hs, :Tm))
        Hg = ncread(g, "Hs")[:, :, 1]
        wetg = m.env.wet[3:end-2, 3:end-2]
        @test Hg[wetg] ≈ Float32.(bulk(m).Hs[wetg]) && all(isnan, Hg[.!wetg])
        @test ncread(g, "x") ≈ xc
        @test_throws ErrorException BulkWriter(joinpath(dirn, "bad.nc"), bm; layout = :other)
        w = BulkWriter(joinpath(dirn, "missing.nc"), bm; vars = (:Hs, :Tm))
        @test_throws ErrorException write_fields!(w, 0.0; Hs = lin)
        close(w)
    end
end
