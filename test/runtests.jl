# Unit tests for the BioKan core (src/). Run with:
#   julia --project=. -e 'using Pkg; Pkg.test()'
#   julia --project=. test/runtests.jl

using Test, Random
include(joinpath(@__DIR__, "..", "src", "BioKan.jl"))
using .BioKan
using Catalyst

const SPECIES = [:D, :M, :I, :E_int, :T, :E_ext, :C, :O, :S,
                 :D_diff, :E_ext_diff, :C_diff, :O_diff, :mRNA_M, :mRNA_S]

# Build the circuit once and share it across bacteria/networks, as the
# experiments do.
const CIRCUIT, DEFAULTS = BioKan.create_hebbian_non_spike_model(:node)

make_u0() = BioKan.map_symbols_to_species(CIRCUIT, Dict(sp => 0.0 for sp in SPECIES))
make_params() = Dict(parameters(CIRCUIT) .=> DEFAULTS)
make_bacterium(id, pos) = BioKan.Bacterium(id, pos, CIRCUIT, make_params(), make_u0(); mode=:ssa)

@testset "BioKan" begin

    @testset "Hebbian non-spike circuit" begin
        @test length(parameters(CIRCUIT)) == 43
        @test length(DEFAULTS) == 43
        @test length(species(CIRCUIT)) == length(SPECIES)

        pnames = Symbol.(string.(parameters(CIRCUIT)))
        @test pnames[1:3] == [:n, :m, :l]      # experiments rely on l being 3rd
        @test :k_deg_D in pnames
        @test :k_deg_S in pnames
    end

    @testset "Bacterium lifecycle" begin
        b = make_bacterium(1, [0.0, 0.0])

        BioKan.set_species!(b, :S, 50.0)
        @test BioKan.get_species(b, :S) == 50.0

        BioKan.set_species!(b, :D, 100.0)
        BioKan.notify_bacterium!(b)
        BioKan.step_bacterium!(b, 1.0)

        for sp in SPECIES
            v = BioKan.get_species(b, sp)
            @test isfinite(v)
            @test v >= 0.0                     # SSA molecule counts stay non-negative
        end
    end

    @testset "Network construction and edges" begin
        d_max = 1.2e-5
        s = 1e-5
        net = BioKan.BioNetwork(d_max, 3)
        for (i, pos) in enumerate(([-s, 0.0], [s, 0.0], [0.0, 0.0]))
            BioKan.add_bacterium!(net, make_bacterium(i, pos))
        end
        BioKan.build_edges!(net)

        @test length(net.nodes) == 3
        @test !isempty(net.edges)

        # B1(-s,0)–B2(+s,0) are 2s apart (> d_max) so must NOT be directly coupled,
        # while both sit at distance s (< d_max) from B3(0,0).
        neighbours(i) = Set(id2 for (id1, id2, _) in net.edges if id1 == i && id1 != id2)
        @test 3 in neighbours(1)
        @test 3 in neighbours(2)
        @test !(2 in neighbours(1))
    end

    @testset "Static coupling and instantaneous transport" begin
        d_max = 1.2e-5
        s = 1e-5
        R_cell = 0.5e-6
        dt = 1.0
        gamma = 0.02

        net = BioKan.BioNetwork(d_max, 2)
        BioKan.add_bacterium!(net, make_bacterium(1, [0.0, 0.0]))
        BioKan.add_bacterium!(net, make_bacterium(2, [s, 0.0]))
        BioKan.build_edges!(net)

        D_dict = Dict(sp => (sp == :D_diff ? (0.15 * s)^2 * gamma : 0.0) for sp in SPECIES)
        gamma_dict = Dict(sp => (sp == :D_diff ? gamma : 0.0) for sp in SPECIES)
        weights = BioKan.compute_static_coupling_physics(
            net.edges, D_dict, gamma_dict, SPECIES, R_cell, dt)

        n_bac = 2
        n_species = length(SPECIES)
        di = findfirst(==(:D_diff), SPECIES)

        # Coupling weight is a hitting probability in (0, 1]; self-edges are dropped.
        @test haskey(weights, (1, 2))
        @test 0.0 < weights[(1, 2)][di] <= 1.0
        @test !haskey(weights, (1, 1))

        # Transport is a Binomial draw, so use a large emission and a fixed seed to
        # make the "B1's emission reaches B2" check deterministic.
        Random.seed!(1234)
        flux = zeros(Float64, n_bac, n_species)
        flux[1, di] = 1_000_000.0
        received = zeros(Float64, n_bac, n_species)
        BioKan.propagate_signals_instantaneous!(received, weights, flux, n_bac, n_species)

        @test all(received .>= 0.0)
        @test received[2, di] > 0.0            # B1's emission reaches its neighbour B2
        @test received[1, di] == 0.0           # a cell does not receive its own emission
    end

    @testset "Logic gate truth tables" begin
        # Truth-table order is (none, B_only, A_only, AB).
        @test length(BioKan.LOGIC_GATES) == 16
        @test BioKan.LOGIC_GATES[:XOR]  == [0, 1, 1, 0]
        @test BioKan.LOGIC_GATES[:AND]  == [0, 0, 0, 1]
        @test BioKan.LOGIC_GATES[:OR]   == [0, 1, 1, 1]
        @test BioKan.LOGIC_GATES[:NAND] == [1, 1, 1, 0]
        @test BioKan.LOGIC_GATES[:XNOR] == [1, 0, 0, 1]
    end

    @testset "XOR protocol shape" begin
        out = BioKan.pattern_to_learn_xor(; logical_gate = :XOR, epochs = 1,
                                          test_epochs = 1, pause_epochs = 0, dt = 1.0)
        time_steps = out[1]
        Input_A = out[2]
        Input_B = out[3]
        @test length(Input_A) == length(time_steps)
        @test length(Input_B) == length(time_steps)
        @test all(>=(0.0), Input_A)
    end
end
