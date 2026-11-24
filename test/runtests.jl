using BinWaves
using Test

@testset "BinWaves.jl" begin
    include("test_gse.jl")
    include("test_windkp.jl")
    include("test_blocks.jl")
end
