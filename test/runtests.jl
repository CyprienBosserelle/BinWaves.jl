using BinWaves
using Test

@testset "BinWaves.jl" begin
    include("test_gse.jl")
    include("test_windkp.jl")
end
