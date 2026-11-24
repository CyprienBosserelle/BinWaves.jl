# Tests of the local wind-sea look-up table (build_wind_kp / wind_sea / reconstruct(...; wind)).
using BinWaves
using Test, Printf

# semi-enclosed bay: open to the west, headlands along the north and south sides, beach at the east end
function bay_model(; params = WaveParams(), ns = 12, nth = 24)
    nx, ny, dx = 32, 20, 500.0
    x = collect((0.5:1:nx) .* dx); y = collect((0.5:1:ny) .* dx)
    D = [25.0 - 22.0 * (i - 1) / (nx - 2) for i in 1:nx, j in 1:ny]
    for j in 1:ny, i in 1:nx
        ((j <= 2 || j >= ny - 1) && i >= 6) && (D[i, j] = 0.0)
        i == nx && (D[i, j] = 0.0)
    end
    spec = SpectralGrid(ns = ns, nth = nth, fmin = 0.07, fmax = 0.8)
    return WaveModel(CartesianGrid(x, y); depth = D, spec = spec, params = params)
end

wetcells(m) = m.env.wet[WaveAction.NG+1:WaveAction.NG+m.grid.nx, WaveAction.NG+1:WaveAction.NG+m.grid.ny]

function direct_wind(m, U, d; sweep = SweepParams(w_theta = 0.0, w_sigma = 0.0, dabs = 1e-4, maxiter = 200), bcleft = ZeroBC())
    m.params = BinWaves._with(m.params; wind = true, whitecap = true)
    set_boundaries!(m; left = bcleft, right = ZeroBC(), bottom = ZeroBC(), top = ZeroBC())
    initialize!(m; Hs = 0.0)
    set_wind!(m; U10 = U, dir = d)
    solve_stationary!(m; sweep = sweep, verbose = false)
    return bulk(m).Hs
end

rmsrel(a, b, mask) = sqrt(sum(((a .- b) .^ 2)[mask]) / sum((b .^ 2)[mask]))

@testset "local wind sea (WindKP)" begin
    m = bay_model()
    pts = [(16, 10), (26, 10)]
    p0 = m.params
    t = @elapsed wk = build_wind_kp(m; dirs = 0:45:315, speeds = [4.0, 8.0, 16.0, 24.0], points = pts, verbose = false)
    @printf("   wind table: %d runs in %.1f s\n", length(wk.done), t)
    wet = wetcells(m)

    @testset "model left as it was" begin
        @test m.params == p0
        @test all(b -> b isa OpenBC, m.bc)
        @test all(iszero, m.env.u10x) && all(iszero, m.env.u10y)
    end

    @testset "trained states are reproduced exactly" begin
        ws = wind_sea(wk, [8.0, 16.0], [90.0, 270.0])
        @test ws.m0[:, :, 1] ≈ wk.m0[3, 2, :, :]
        @test ws.m0[:, :, 2] ≈ wk.m0[7, 3, :, :]
        @test ws.point_var[2, :, :, 1] ≈ wk.spec[7, 3, :, :, 1]
        @test ws.Hs[:, :, 2] ≈ 4 .* sqrt.(wk.m0[7, 3, :, :])
        Hd = direct_wind(bay_model(), 16.0, 270.0)
        @test maximum(abs.(ws.Hs[:, :, 2] .- Hd)) < 1e-5
        # a light wind below umin and NaN give no wind sea
        z = wind_sea(wk, [0.5, NaN], [270.0, 270.0])
        @test all(iszero, z.m0) && all(iszero, z.point_var)
    end

    @testset "interpolation in speed and direction vs direct runs" begin
        mask = wet .& (Hd = direct_wind(bay_model(), 12.0, 250.0); Hd .> 0.05)
        ell = rmsrel(wind_sea(wk, [12.0], [250.0]).Hs[:, :, 1], Hd, mask)
        eli = rmsrel(wind_sea(wk, [12.0], [250.0]; interp = :linear).Hs[:, :, 1], Hd, mask)
        @printf("   U = 12 m/s, 250°: rms relative Hs error  log-log %.3f   linear %.3f\n", ell, eli)
        @test ell < 0.08
        # pure speed interpolation (trained direction): isolates the speed curve
        Hd2 = direct_wind(bay_model(), 12.0, 270.0); mask2 = wet .& (Hd2 .> 0.05)
        es = rmsrel(wind_sea(wk, [12.0], [270.0]).Hs[:, :, 1], Hd2, mask2)
        esl = rmsrel(wind_sea(wk, [12.0], [270.0]; interp = :linear).Hs[:, :, 1], Hd2, mask2)
        @printf("   U = 12 m/s, 270° (speed only): log-log %.3f   linear %.3f\n", es, esl)
        @test es < 0.05
        @test es < esl
        # extrapolation beyond the trained range stays bounded and monotonic
        hi = wind_sea(wk, [24.0, 28.0, 32.0], fill(270.0, 3)).m0
        @test all(hi[:, :, 1] .<= hi[:, :, 2] .<= hi[:, :, 3])
        @test all(hi[:, :, 3] .<= hi[:, :, 1] .* (32 / 24)^4 .+ 1f-6)
        lo = wind_sea(wk, [2.0, 3.0, 4.0], fill(270.0, 3)).m0
        @test all(lo[:, :, 1] .<= lo[:, :, 2] .<= lo[:, :, 3])
        # point spectra carry the same variance as the field at the point
        ws = wind_sea(wk, [12.0], [250.0])
        for (q, (i, j)) in enumerate(pts)
            @test sum(ws.point_var[1, :, :, q]) ≈ ws.m0[i, j, 1] rtol = 0.02
        end
        pp = point_parameters(wk, ws.point_var)
        @test pp.Hs[1, 1] ≈ 4sqrt(ws.m0[16, 10, 1]) rtol = 0.01
    end

    @testset "NetCDF storage and checkpoint resume" begin
        dir = mktempdir()
        f = joinpath(dir, "wind.nc")
        save_wind_kp_netcdf(f, wk)
        w2 = load_wind_kp_netcdf(f)
        @test w2.m0 == wk.m0 && w2.spec == wk.spec && w2.done == wk.done && w2.dirs == wk.dirs && w2.speeds == wk.speeds
        @test w2.points == wk.points
        ck = joinpath(dir, "ck.nc")
        mc = bay_model()
        wa = build_wind_kp(mc; dirs = [270.0, 90.0], speeds = [8.0, 16.0], points = pts, verbose = false, checkpoint = ck)
        tr = @elapsed wb = build_wind_kp(mc; dirs = [270.0, 90.0], speeds = [8.0, 16.0], points = pts, verbose = false, checkpoint = ck)
        @test wb.m0 == wa.m0 && all(wb.done)
        @test tr < 5
        @test_throws ErrorException build_wind_kp(mc; dirs = [270.0], speeds = [8.0], verbose = false, checkpoint = ck)
    end

    @testset "swell + local wind sea vs one direct run" begin
        # swell coefficients on the same model (wind off), only for the cases that carry energy
        ms = bay_model()
        f = collect(range(0.04, 0.6, length = 80)); dirs = collect(0.0:5.0:355.0)
        E = zeros(1, length(f), length(dirs))
        Hs0, Tp0, d0 = 1.0, 8.0, 260.0
        for (j, d) in enumerate(dirs), (i, fi) in enumerate(f)
            fp = 1 / Tp0; sg = fi <= fp ? 0.07 : 0.09
            S = fi^-5 * exp(-1.25 * (fp / fi)^4) * 3.3^exp(-(fi - fp)^2 / (2sg^2 * fp^2))
            dd = deg2rad(mod(d - d0 + 180, 360) - 180)
            E[1, i, j] = S * (abs(dd) < π / 2 ? cos(dd)^10 : 0.0)
        end
        E .*= (Hs0^2 / 16) / (sum(E) * (f[2] - f[1]) * 5.0)
        kp0 = build_kp(ms; sides = (:left,), points = pts, cases = Int[], verbose = false)
        V, cap = bin_offshore_spectrum(kp0, E, f, dirs)
        use = findall(V[1, :] .> 1e-5 * maximum(V))
        kp = build_kp(ms; sides = (:left,), points = pts, cases = use, verbose = false)
        kp.done .= true                                   # the other cases carry no energy for this spectrum
        # the same spectrum as an explicit boundary condition for the direct run
        sp = ms.spec
        Nb = zeros(sp.ns, sp.nth)
        for (c, (a, h)) in enumerate(kp.cases)
            is, it = only(kp.fgroups[a]), only(kp.dgroups[h])
            Nb[is, it] = V[1, c] / (sp.sig[is] * sp.dsig[is] * sp.dth)
        end
        Hsw = reconstruct(kp, E, f, dirs).Hs[:, :, 1]
        Hsd = (mm = bay_model(); set_boundaries!(mm; left = SpectrumBC(Nb), right = ZeroBC(), bottom = ZeroBC(), top = ZeroBC());
               initialize!(mm; Hs = 0.0); solve_stationary!(mm; sweep = SweepParams(w_theta = 0.0, w_sigma = 0.0, dabs = 1e-4),
               verbose = false); bulk(mm).Hs)
        rec = reconstruct(kp, E, f, dirs; wind = wk, U10 = [12.0], wdir = [250.0])
        Hw = wind_sea(wk, [12.0], [250.0]).Hs[:, :, 1]
        @test rec.Hs[:, :, 1] .^ 2 ≈ Hsw .^ 2 .+ Hw .^ 2 rtol = 1e-4
        @test rec.wind.m0[:, :, 1] ≈ Hw .^ 2 ./ 16
        @test rec.point_var[1, :, :, :] ≈ reconstruct(kp, E, f, dirs).point_var[1, :, :, :] .+ rec.wind.point_var[1, :, :, :]
        Hc = direct_wind(bay_model(), 12.0, 250.0; bcleft = SpectrumBC(Nb))
        mask = wet .& (Hc .> 0.05)
        e_sw = rmsrel(Hsw, Hsd, mask)
        e_all = rmsrel(rec.Hs[:, :, 1], Hc, mask)
        @printf("   swell only: rms rel. error %.3f;   swell + wind sea (12 m/s, 250°) vs coupled direct run: %.3f\n", e_sw, e_all)
        @test e_sw < 0.02
        @test e_all < 0.15
        # error of simply ignoring the local wind sea, for reference
        @printf("   (ignoring the wind sea: %.3f)\n", rmsrel(Hsw, Hc, mask))
    end
end
