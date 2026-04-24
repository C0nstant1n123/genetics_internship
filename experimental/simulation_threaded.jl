# ==============================================================================
# experimental/simulation_threaded.jl
#
# Version expérimentale de run_hebbian_simulation_full avec parallélisation
# de l'étape B (step_bacterium!) via Threads.@threads.
#
# DIFFÉRENCE vs simulation.jl :
#   - Étape B utilise Threads.@threads pour distribuer les bactéries
#     sur les threads Julia disponibles.
#   - Tout le reste est identique à run_hebbian_simulation_full.
#
# LIMITATION CONNUE :
#   - step_bacterium! modifie l'état interne de chaque Bacterium.
#     Les bactéries sont indépendantes à cette étape → pas de race condition.
#     Mais si step_bacterium! utilise un RNG global, il faut vérifier que
#     JumpProcesses utilise bien des RNG par-thread (c'est le cas depuis Julia 1.7+).
#
# USAGE :
#   julia --project=. --threads=4 experimental/simulation_threaded.jl
#
# Pour comparer avec la version séquentielle, lancer les deux et comparer
# le temps affiché.
# ==============================================================================

using Pkg, Printf, Statistics
PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT)

using Catalyst, JumpProcesses, OrdinaryDiffEq,
      Statistics, Random, Distributions, LinearAlgebra

include(joinpath(PROJECT_ROOT, "src",     "BioKan.jl"))
include(joinpath(PROJECT_ROOT, "scripts", "simulation.jl"))
using .BioKan
using .BioSim

println("Threads disponibles : $(Threads.nthreads())")

# ==============================================================================
# FONCTION THREADÉE
# Identique à run_hebbian_simulation_full, seule l'étape B change.
# ==============================================================================
function run_hebbian_simulation_threaded(θ::Vector{Float64},
                                         gate_name::Symbol,
                                         cfg::BioSim.HebbianConfig)

    n_bacteries       = cfg.n_bacteries
    dt                = cfg.dt
    R_cell            = cfg.R_cell
    species_names     = cfg.species_names
    n_species         = length(species_names)
    D_dict            = cfg.D_dict
    gamma_dict        = cfg.gamma_dict
    diffusion_targets = cfg.diffusion_targets

    time_steps, Input_A, Input_B, Output_0, Output_1, mask =
        BioKan.pattern_to_learn_density(gate_name)
    total_steps = length(time_steps)

    circuit, _ = BioKan.create_hebbian_model(:hebbian_node)
    params_dict = Dict(parameters(circuit) .=> θ)
    u0_raw  = Dict(s => 0.0 for s in species_names)
    u0_dict = BioKan.map_symbols_to_species(circuit, u0_raw)

    net = BioKan.BioNetwork(cfg.d_max, n_bacteries)
    BioKan.build_network_cube!(net, n_bacteries, cfg.d_max, cfg.n_segments,
                               circuit, params_dict, u0_dict, :ssa)
    node_roles = BioKan.assign_tetrahedral_roles(net)

    weights = BioKan.compute_static_coupling_physics(
        net.edges, D_dict, gamma_dict, species_names, R_cell, dt)

    GC.gc()
    flux_emissions = zeros(Float64, n_bacteries, n_species)
    history        = zeros(Float32, total_steps, n_bacteries, n_species)

    # Pré-calculer la liste ordonnée des ids pour Threads.@threads
    # (les Dict ne garantissent pas l'ordre)
    ids = collect(1:n_bacteries)

    for step in 1:total_steps

        # A. INPUTS — séquentiel (écritures sur des bactéries spécifiques)
        BioKan.inject_pattern_step!(net, step,
                                    Input_A, Input_B, Output_0, Output_1,
                                    node_roles)

        # B. BIOLOGIE INTERNE — threadé
        # Chaque bactérie est indépendante à cette étape.
        Threads.@threads for i in ids
            haskey(net.nodes, i) || continue
            BioKan.step_bacterium!(net.nodes[i], dt)
        end

        # C. FLUX — séquentiel
        fill!(flux_emissions, 0.0)
        for i in ids
            haskey(net.nodes, i) || continue
            b = net.nodes[i]
            for s in 1:n_species
                sym = species_names[s]
                D_dict[sym] > 1e-40 &&
                    (flux_emissions[i, s] = max(0.0, BioKan.get_species(b, sym)))
            end
        end

        # D. TRANSPORT — séquentiel (opération globale)
        received = BioKan.propagate_signals_instantaneous!(
            weights, flux_emissions, n_bacteries, n_species)

        # E. UPDATE — séquentiel (dépend de received)
        for i in ids
            haskey(net.nodes, i) || continue
            b = net.nodes[i]
            for s in 1:n_species
                sym = species_names[s]
                if D_dict[sym] > 1e-40
                    gamma  = gamma_dict[sym]
                    target = diffusion_targets[sym]
                    amt    = max(0.0, received[i, s] * exp(-gamma * dt))
                    BioKan.set_species!(b, target, BioKan.get_species(b, target) + amt)
                    BioKan.set_species!(b, sym, 0.0)
                end
                history[step, i, s] = max(0.0f0, Float32(BioKan.get_species(b, sym)))
            end
        end
    end

    d_idx = findfirst(==(:D), species_names)
    loss  = BioKan.compute_loss(history, time_steps, mask, node_roles, d_idx, θ[31])
    return loss
end

# ==============================================================================
# BENCHMARK : séquentiel vs threadé
# ==============================================================================
_, θ₀ = BioKan.create_hebbian_model(:node_generic)
cfg   = BioSim.default_hebbian_config()

println("\n--- Séquentiel ---")
t1 = time()
loss_seq = BioSim.run_hebbian_simulation(θ₀, :XOR, cfg)
t_seq = time() - t1
@printf "Loss : %.4f | Durée : %.1f s\n" loss_seq t_seq

println("\n--- Threadé ($(Threads.nthreads()) threads) ---")
t2 = time()
loss_thr = run_hebbian_simulation_threaded(θ₀, :XOR, cfg)
t_thr = time() - t2
@printf "Loss : %.4f | Durée : %.1f s\n" loss_thr t_thr

@printf "\nAccélération : ×%.2f\n" (t_seq / t_thr)
