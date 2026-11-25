# Tests of the block-adaptive mesh (WaveBlocks) and of BinWaves on it.
using BinWaves
using Test, Printf, NetCDF

const NGb = WaveAction.NG

function uniform_model(depth, Lx, Ly, dx; spec, params)
    xc = collect((0.5:1:round(Int, Lx / dx)) .* dx); yc = collect((0.5:1:round(Int, Ly / dx)) .* dx)
    m = WaveModel(CartesianGrid(xc, yc); depth = [max(depth(x, y), 0.0) for x in xc, y in yc], spec = spec, params = params)
    return m, xc, yc
end

@testset "block-adaptive mesh (WaveBlocks)" begin
    spec = SpectralGrid(ns = 10, nth = 24, fmin = 0.05, fmax = 0.4)

    @testset "mesh: refinement, 2:1 balance, lookup" begin
        mesh = BlockMesh(; Lx = 6400.0, Ly = 3200.0, dx0 = 100.0, bs = 16, maxlevel = 3)
        @test nblocks(mesh) == 8 && ncells(mesh) == 8 * 256
        refine!(mesh) do k               # one block at the far corner, three levels down: forces balancing around it
            xlo, _, ylo, _ = block_bounds(mesh, k); xlo == 0 && ylo == 0
        end
        refine!(mesh) do k; block_bounds(mesh, k)[1:3:4] == (0.0, 800.0); end
        refine!(mesh) do k; block_bounds(mesh, k)[1:3:4] == (0.0, 400.0); end
        @test maximum(mesh.level) == 3
        for k in 1:nblocks(mesh), s in (:left, :right, :bottom, :top)
            @test all(n -> abs(mesh.level[n] - mesh.level[k]) <= 1, WaveBlocks.face_neighbours(mesh, k, s))
        end
        @test sum(4.0 .^ (-mesh.level)) ≈ 8                                      # the blocks tile the domain
        x, y = cell_centres(mesh); h = cell_dx(mesh)
        @test sum(h .^ 2) ≈ 6400 * 3200
        for c in (1, 300, ncells(mesh) ÷ 2, ncells(mesh))
            k, i, j = locate(mesh, x[c], y[c])
            @test cell_index(mesh, k, i, j) == c
        end
        @test locate(mesh, -1.0, 10.0) === nothing
        # refinement by depth: shallow water and the coastline get the finest cells
        dep(x, y) = 30.0 * (1 - x / 6000)
        m2 = BlockMesh(; Lx = 6400.0, Ly = 3200.0, dx0 = 100.0, maxlevel = 2)
        refine_by_depth!(m2, dep; resolution = d -> 6d)
        for k in 1:nblocks(m2)
            xlo, xhi, _, _ = block_bounds(m2, k)
            dmin = dep(xhi, 0.0)
            dmin > 0.05 && @test block_dx(m2, m2.level[k]) <= max(6dmin, 25.0) + 1e-9
            xlo > 6000 && @test m2.level[k] == 0                                 # dry blocks are not refined
        end
        # raster input
        xb = collect(0:50.0:6400); yb = collect(0:50.0:3200)
        m3 = BlockMesh(; Lx = 6400.0, Ly = 3200.0, dx0 = 100.0, maxlevel = 2)
        refine_by_depth!(m3, (xb, yb, [dep(a, b) for a in xb, b in yb]); resolution = d -> 6d)
        @test m3.level == m2.level
        # regular-grid sampling
        G = to_grid(m2, Float64.(1:ncells(m2)), [10.0, 6390.0], [10.0])
        @test G[1] == cell_index(m2, locate(m2, 10.0, 10.0)...) && G[2] == cell_index(m2, locate(m2, 6390.0, 10.0)...)
    end

    @testset "uniform mesh reproduces the single-grid model exactly" begin
        depth(x, y) = max(0.0, 15.0 - 14.0 * x / 2400) - 3exp(-((x - 1200)^2 + (y - 800)^2) / 300^2)
        prm = WaveParams(breaking = true, friction = true)
        mesh = BlockMesh(; Lx = 2400.0, Ly = 1600.0, dx0 = 50.0, maxlevel = 1)
        bm = BlockModel(mesh; depth = depth, spec = spec, params = prm)
        m, xc, yc = uniform_model(depth, 2400.0, 1600.0, 50.0; spec = spec, params = prm)
        bc = SpectrumBC(Hs = 1.5, Tp = 9.0, dir = 250.0)
        set_boundaries!(m; left = bc, bottom = ZeroBC()); set_boundaries!(bm; left = bc, bottom = ZeroBC())
        idx = grid_index(mesh, xc, yc)
        @test to_grid(cell_depth(bm), idx) ≈ m.env.d[NGb+1:end-NGb, NGb+1:end-NGb]
        @test stable_dt(bm) == stable_dt(m)
        dt = stable_dt(m)
        run!(m; tend = 30dt, dt = dt, verbose = false); run!(bm; tend = 30dt, dt = dt, verbose = false)
        @test maximum(abs.(to_grid(bulk(bm).Hs, idx) .- bulk(m).Hs)) < 1e-12
        @test total_action(bm) ≈ total_action(m) rtol = 1e-12
        run_implicit!(m; tend = 300.0, dt = 30.0, verbose = false); run_implicit!(bm; tend = 300.0, dt = 30.0, verbose = false)
        @test maximum(abs.(to_grid(bulk(bm).Hs, idx) .- bulk(m).Hs)) < 1e-12
        sw = SweepParams(w_theta = 0.0, w_sigma = 0.0, maxiter = 100)
        initialize!(m); initialize!(bm)
        set_gse!(m; alpha_s = 3.0, alpha_n = 3.0); set_gse!(bm; alpha_s = 3.0, alpha_n = 3.0)
        itm = solve_stationary!(m; sweep = sw, verbose = false); itb = solve_stationary!(bm; sweep = sw, verbose = false)
        @test itm == itb
        @test maximum(abs.(to_grid(bulk(bm).Hs, idx) .- bulk(m).Hs)) < 1e-12
        @test maximum(abs.(to_grid(bulk(bm).Dir, idx) .- bulk(m).Dir)) < 1e-9
    end

    @testset "refinement interfaces: uniform deep-water swell" begin
        mesh = BlockMesh(; Lx = 4800.0, Ly = 3200.0, dx0 = 100.0, maxlevel = 2)
        refine!(mesh) do k; (a, b, c, d) = block_bounds(mesh, k); a >= 1600 && b <= 3200 && c >= 1600; end
        refine!(mesh) do k; (a, b, c, d) = block_bounds(mesh, k); a >= 2400 && b <= 3200 && c >= 1600 && d <= 2400; end
        @test sort(unique(mesh.level)) == [0, 1, 2]
        bm = BlockModel(mesh; depth = 200.0, spec = spec, params = WaveParams(refraction = true))
        bc = SpectrumBC(Hs = 1.0, Tp = 8.0, dir = 270.0, spread = 5.0)
        set_boundaries!(bm; left = bc, bottom = OpenBC(), top = OpenBC())
        it, conv = solve_stationary!(bm; sweep = SweepParams(w_theta = 0.0, w_sigma = 0.0, maxiter = 100), verbose = false)
        @test conv
        Hs = bulk(bm).Hs
        @printf("   uniform swell through level 0/1/2 interfaces: Hs in [%.5f, %.5f] m (%d iterations)\n", extrema(Hs)..., it)
        @test maximum(Hs) - minimum(Hs) < 1e-3
        @test abs(Hs[1] - 1.0) < 0.01                    # the spectral grid truncates the JONSWAP tail
        # explicit: the same, and action is nearly conserved across the interfaces (closed by OpenBC on all sides)
        bm2 = BlockModel(mesh; depth = 200.0, spec = spec)
        initialize!(bm2; Hs = 1.0, Tp = 8.0, dir = 270.0)
        H0 = bulk(bm2).Hs[1]
        run!(bm2; tend = 60.0, verbose = false)
        @test maximum(abs.(bulk(bm2).Hs .- H0)) < 1e-9                        # a uniform state stays uniform
    end

    @testset "adaptive mesh vs uniform fine grid (beach with a shoal)" begin
        Lx, Ly = 3200.0, 1600.0
        depth(x, y) = 20.0 * (1 - x / 3000) - 4exp(-((x - 2000)^2 + (y - 800)^2) / 250^2)
        prm = WaveParams(breaking = true, friction = true)
        bc = SpectrumBC(Hs = 1.5, Tp = 9.0, dir = 255.0, spread = 20.0)
        sw = SweepParams(w_theta = 0.0, w_sigma = 0.0, maxiter = 150)
        mesh = BlockMesh(; Lx = Lx, Ly = Ly, dx0 = 50.0, maxlevel = 2)
        refine_by_depth!(mesh, depth; resolution = d -> 4d)
        bm = BlockModel(mesh; depth = depth, spec = spec, params = prm)
        set_boundaries!(bm; left = bc)
        tb = @elapsed solve_stationary!(bm; sweep = sw, verbose = false)
        mf, xf, yf = uniform_model(depth, Lx, Ly, 12.5; spec = spec, params = prm)
        set_boundaries!(mf; left = bc)
        tf = @elapsed solve_stationary!(mf; sweep = sw, verbose = false)
        Hf = bulk(mf).Hs
        Hb = to_grid(bulk(bm).Hs, grid_index(mesh, xf, yf))
        wet = Hf .> 0.05
        e = sqrt(sum(((Hb .- Hf) .^ 2)[wet]) / count(wet))
        @printf("   %s\n   rms ΔHs vs uniform 12.5 m grid: %.4f m (max %.3f), cells %d vs %d, time %.1f s vs %.1f s\n",
                mesh_summary(mesh), e, maximum(abs.(Hb .- Hf)[wet]), ncells(mesh), length(Hf), tb, tf)
        @test ncells(mesh) < 0.7 * length(Hf)
        @test e < 0.01
    end

    @testset "BinWaves coefficients and wind table on a block mesh" begin
        Lx, Ly = 3200.0, 1600.0
        depth(x, y) = 20.0 * (1 - x / 3000) - 4exp(-((x - 2000)^2 + (y - 800)^2) / 250^2)
        mesh = BlockMesh(; Lx = Lx, Ly = Ly, dx0 = 100.0, maxlevel = 2)
        refine_by_depth!(mesh, depth; resolution = d -> 8d)
        bm = BlockModel(mesh; depth = depth, spec = spec)
        pts = [(1000.0, 800.0), (2600.0, 700.0)]
        # offshore spectrum on its own grid
        f = collect(range(0.04, 0.45, length = 60)); dirs = collect(0.0:5.0:355.0)
        E = zeros(1, length(f), length(dirs))
        for (j, d) in enumerate(dirs), (i, fi) in enumerate(f)
            fp = 1 / 9; sg = fi <= fp ? 0.07 : 0.09
            dd = deg2rad(mod(d - 255 + 180, 360) - 180)
            E[1, i, j] = fi^-5 * exp(-1.25 * (fp / fi)^4) * 3.3^exp(-(fi - fp)^2 / (2sg^2 * fp^2)) * (abs(dd) < π / 2 ? cos(dd)^8 : 0.0)
        end
        E .*= (1.2^2 / 16) / (sum(E) * (f[2] - f[1]) * 5.0)
        kp0 = build_kp(bm; sides = (:left,), points = pts, cases = Int[], verbose = false)
        V, _ = bin_offshore_spectrum(kp0, E, f, dirs)
        use = findall(V[1, :] .> 1e-5 * maximum(V))
        kp = build_kp(bm; sides = (:left,), points = pts, cases = use, verbose = false)
        kp.done .= true
        @test size(kp.kp_m0) == (length(kp.cases), ncells(mesh), 1)
        @test haskey(kp.info, "mesh")
        rec = reconstruct(kp, E, f, dirs)
        @test size(rec.Hs) == (ncells(mesh), 1, 1)
        # direct run with the same boundary spectrum
        sp = bm.spec
        Nb = zeros(sp.ns, sp.nth)
        for (c, (a, h)) in enumerate(kp.cases)
            is, it = only(kp.fgroups[a]), only(kp.dgroups[h])
            Nb[is, it] = V[1, c] / (sp.sig[is] * sp.dsig[is] * sp.dth)
        end
        initialize!(bm)
        set_boundaries!(bm; left = SpectrumBC(Nb), right = ZeroBC(), bottom = ZeroBC(), top = ZeroBC())
        solve_stationary!(bm; sweep = SweepParams(w_theta = 0.0, w_sigma = 0.0, dabs = 1e-4), verbose = false)
        Hd = bulk(bm).Hs; wet = Hd .> 0.05
        e = sqrt(sum(((rec.Hs[:, 1, 1] .- Hd) .^ 2)[wet]) / sum((Hd .^ 2)[wet]))
        @printf("   superposition on the block mesh: rms relative Hs error %.2e\n", e)
        @test e < 0.01
        # direction and spread at every mesh cell
        bdir = bulk(bm).Dir
        m0 = zeros(ncells(mesh), 1); a = zeros(ncells(mesh), 1); b = zeros(ncells(mesh), 1)
        BinWaves._moments!(m0, a, b, bm, 1.0)
        spr = [BinWaves._dir_spread(m0[c], a[c], b[c])[2] for c in 1:ncells(mesh)]
        cd(x, y) = abs(mod(x - y + 180, 360) - 180)
        ed = maximum(cd.(rec.Dir[wet, 1, 1], bdir[wet])); es = maximum(abs.(rec.Spread[wet, 1, 1] .- spr[wet]))
        @printf("   direction on the block mesh: max error %.2e°, spread %.2e°\n", ed, es)
        @test ed < 0.01 && es < 0.01
        # CF output of the reconstructed direction (circular interpolation on the finest grid)
        fo = joinpath(mktempdir(), "rec.nc")
        save_fields_netcdf(fo, kp, [0.0]; layout = :both, Hs = rec.Hs, Dir = rec.Dir, Spread = rec.Spread)
        @test ncgetatt(fo, "Spread", "standard_name") == "sea_surface_wave_directional_spread"
        # coefficients without the direction moments: smaller, no Dir / Spread, files still load
        kpn = build_kp(bm; sides = (:left,), points = pts, cases = use[1:2], verbose = false, dir_moments = false)
        @test size(kpn.kp_cos) == (length(kpn.cases), 0, 0)
        kpn.done .= true                                   # only two cases computed: silence the incomplete-build warnings
        fn = joinpath(mktempdir(), "kpn.nc"); save_kp_netcdf(fn, kpn)
        kpn2 = load_kp_netcdf(fn)
        @test size(kpn2.kp_cos, 2) == 0
        rn = reconstruct(kpn2, E, f, dirs)
        @test size(rn.Dir) == (0, 0, 0) && size(rn.Hs) == (ncells(mesh), 1, 1)
        pp = point_parameters(kp, rec.point_var)
        for (q, (x, y)) in enumerate(pts)
            @test pp.Hs[1, q] ≈ Hd[cell_index(mesh, locate(mesh, x, y)...)] rtol = 0.01
        end
        # NetCDF round trip keeps the mesh
        dir = mktempdir()
        save_kp_netcdf(joinpath(dir, "kp.nc"), kp)
        kp2 = load_kp_netcdf(joinpath(dir, "kp.nc"))
        @test kp2.info["mesh"].level == mesh.level && kp2.info["mesh"].bx == mesh.bx && kp2.info["mesh"].by == mesh.by
        @test kp2.x ≈ kp.x && kp2.y ≈ kp.y && kp2.kp_m0 == kp.kp_m0 && kp2.kp_cos == kp.kp_cos && kp2.kp_sin == kp.kp_sin &&
              kp2.points == kp.points
        @test reconstruct(kp2, E, f, dirs).Hs ≈ rec.Hs
        G = to_grid(kp2, rec.Hs, [1000.0, 2600.0], [800.0, 700.0])
        @test G[1, 1] ≈ rec.Hs[cell_index(mesh, locate(mesh, 1000.0, 800.0)...), 1, 1]
        # wind table on the mesh
        wk = build_wind_kp(bm; dirs = [270.0, 90.0], speeds = [8.0, 16.0], points = pts, verbose = false)
        @test size(wk.m0) == (2, 2, ncells(mesh), 1)
        @test all(iszero, bm.blocks[1].env.u10x) && !bm.params.wind && all(b -> !b.params.wind, bm.blocks)
        save_wind_kp_netcdf(joinpath(dir, "w.nc"), wk)
        wk2 = load_wind_kp_netcdf(joinpath(dir, "w.nc"))
        @test wk2.m0 == wk.m0 && haskey(wk2.info, "mesh")
        rw = reconstruct(kp, E, f, dirs; wind = wk, U10 = [16.0], wdir = [270.0])
        @test rw.Hs[:, 1, 1] .^ 2 ≈ rec.Hs[:, 1, 1] .^ 2 .+ 16 .* wk.m0[2, 2, :, 1] rtol = 1e-4   # dirs sorted: [90, 270]
    end
end
