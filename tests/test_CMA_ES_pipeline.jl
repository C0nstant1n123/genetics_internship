# ==============================================================================
# test_pipeline.jl — Validation du pipeline Hebbian end-to-end
#
# Tourne UNE simulation avec les paramètres par défaut et affiche la loss.
# Permet de vérifier que tout compile et s'exécute avant de lancer CMA-ES.
#
# Usage :
#   julia --project=. scripts/test_pipeline.jl
# ==============================================================================

using Pkg
PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT)

using Catalyst, JumpProcesses, OrdinaryDiffEq, Statistics, Random, Distributions, LinearAlgebra

include(joinpath(PROJECT_ROOT, "src",     "BioKan.jl"))
include(joinpath(PROJECT_ROOT, "scripts", "simulation.jl"))
using .BioKan
using .BioSim

println("=== Test pipeline Hebbian ===\n")

# --- 1. Paramètres par défaut ---
_, θ₀ = BioKan.create_hebbian_model(:node_generic)
cfg   = BioSim.default_hebbian_config()

println("Réseau       : $(cfg.n_bacteries) bactéries")
println("Paramètres   : $(length(θ₀))")
println("Gate         : XOR\n")

# --- 2. Une simulation complète ---
println("Lancement de la simulation...")
t_start = time()

loss = BioSim.run_hebbian_simulation(θ₀, :XOR, cfg)

t_elapsed = time() - t_start

println("\n--- Résultat ---")
@printf "Loss        = %.4f  (réf. aléatoire = %.4f)\n" loss log(2)
@printf "Durée       = %.1f s\n" t_elapsed

if loss < log(2)
    println("Le réseau fait mieux qu'un classifieur aléatoire.")
else
    println("Le réseau ne fait pas mieux qu'un classifieur aléatoire (attendu avec θ par défaut).")
end
