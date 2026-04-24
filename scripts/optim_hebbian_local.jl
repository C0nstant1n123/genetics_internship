# ==============================================================================
# optim_hebbian_local.jl — Version locale allégée pour tester CMA-ES
#
# Différences vs optim_hebbian.jl :
#   - Pas de Distributed (séquentiel, 1 worker)
#   - 5 générations seulement
#   - Réseau réduit (8 bactéries via default_hebbian_config)
#   - Pas de sauvegarde JLD2, juste la loss affichée
#
# Usage :
#   julia --project=. scripts/optim_hebbian_local.jl
# ==============================================================================

using Pkg, Printf, Dates, Statistics
PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT)

using Catalyst, JumpProcesses, OrdinaryDiffEq,
      Statistics, Random, Distributions, LinearAlgebra, Evolutionary

include(joinpath(PROJECT_ROOT, "src",     "BioKan.jl"))
include(joinpath(PROJECT_ROOT, "scripts", "simulation.jl"))
using .BioKan
using .BioSim

# ==============================================================================
# CONFIG
# ==============================================================================
const GATE_NAME     = :XOR
const N_GENERATIONS = 4    # générations réelles
const LAMBDA        = 3  # candidats/génération (réduit pour test local)
const SIGMA0        = 0.3   # en espace log : exp(±0.3) ≈ ×0.74/×1.35 par dimension
const CFG           = BioSim.default_hebbian_config()

const PARAM_LO = Float64[2.0,0.1,0.001,0.01,0.01,0.01,0.01,1e-6,10.0,5.0,10.0,5.0,2.0,5.0,10.0,10.0,10.0,10.0,5.0,0.5,0.1,1e-3,1e-5,1e-3,1e-3,1e-3,1e-3,0.1,0.001,0.001,10.0,1e-3]
const PARAM_HI = Float64[20.0,10.0,2.0,5.0,5.0,5.0,5.0,1e-2,500.0,200.0,500.0,200.0,200.0,500.0,500.0,500.0,500.0,500.0,200.0,5.0,0.5,1.0,1e-2,1.0,0.5,1.0,0.5,5.0,1.0,1.0,500.0,1.0]

function evaluate_candidate(φ::Vector{Float64})::Float64
    # φ est en espace log → on décode avant simulation
    θ_c = clamp.(exp.(φ), PARAM_LO, PARAM_HI)
    try
        loss = BioSim.run_hebbian_simulation(θ_c, GATE_NAME, CFG)
        if !isfinite(loss)
            @printf "  [WARN] loss non-finie : %s\n" string(loss)
        end
        return loss
    catch e
        @printf "  [ERROR] simulation crash : %s\n" string(e)
        return Inf
    end
end

# ==============================================================================
# CMA-ES
# ==============================================================================
_, θ₀ = BioKan.create_hebbian_model(:node_generic)
φ₀ = log.(θ₀)   # point de départ en espace log

println("\nCMA-ES local — $(N_GENERATIONS) générations")
println("  gate    = $GATE_NAME")
println("  params  = $(length(θ₀))\n")

result = Evolutionary.optimize(
    evaluate_candidate,
    φ₀,
    CMAES(sigma0 = SIGMA0, lambda = LAMBDA),
    Evolutionary.Options(
        iterations      = N_GENERATIONS,
        abstol          = 0.0,
        reltol          = 0.0,
        show_trace      = true,
        show_every      = 1,
        parallelization = :thread,
    )
)

loss_final = Evolutionary.minimum(result)
@printf "\n--- Résultat ---\n"
@printf "Loss finale  : %.4f\n" loss_final
@printf "Réf. aléat.  : %.4f\n" log(2)
@printf "Amélioration : %s\n" (loss_final < log(2) ? "OUI" : "non")
