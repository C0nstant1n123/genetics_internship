# ==============================================================================
# optim_hebbian.jl — Optimisation CMA-ES du modèle Hebbian bactérien
# ==============================================================================
#
# Stratégie :
#   - Boucle externe de restarts aléatoires (paysage plat → exploration aveugle)
#   - Chaque restart lance CMA-ES pour GENS_PER_RESTART générations
#   - Si loss < SIGNAL_THRESHOLD : signal détecté → bascule en exploitation longue
#   - Sinon : nouveau θ aléatoire dans [PARAM_LO, PARAM_HI]
#   - Meilleur résultat global sauvegardé à chaque checkpoint
#
# Parallélisation :
#   - Distributed : 14 workers (= λ) sur 2 nœuds SLURM via ClusterManagers
#   - Threads     : 4 threads/worker pour paralléliser les bactéries (step_bacterium!)
#   - Chaque génération CMA-ES = 14 simulations en parallèle
#
# Usage :
#   sbatch lancer-optim-hebbian.sh
# ==============================================================================

println("peut etre que ca commence ...")
using Pkg, JLD2, Printf, Dates, Statistics
println("Ca commence...")
PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT)

PATH_BIOKAN = joinpath(PROJECT_ROOT, "src",     "BioKan.jl")
PATH_BIOSIM = joinpath(PROJECT_ROOT, "scripts", "simulation.jl")
OUTPUT_DIR  = joinpath(PROJECT_ROOT, "outputs", "optim_hebbian")
println("La ca commence vraiment...")
# ==============================================================================
# 1. CHARGEMENT
# ==============================================================================
# Parallélisation : Threads.@threads via parallelization = :thread dans Evolutionary
# λ candidats évalués en parallèle sur les threads disponibles (SLURM : --threads=14)
# Pas de Distributed : CMA-ES dans Evolutionary.jl v0.11 ne supporte pas :distributed

using Catalyst, JumpProcesses, OrdinaryDiffEq,
      Statistics, Random, Distributions, LinearAlgebra


println("la ca commence vraiment vraiment...")
include(PATH_BIOKAN)
include(PATH_BIOSIM)
using .BioKan
using .BioSim

println("Threads disponibles : $(Threads.nthreads())")

# ==============================================================================
# 2. CONSTANTES
# ==============================================================================
begin

    const GATE_NAME  = :XOR
    const N_REPLICAS = 1  # Nombre de simulations par évaluation (pour réduire la variance du signal)
    const CFG = BioSim.default_hebbian_config()

    # Point de départ : θ₀ validé dans test_pipeline
    const THETA0 = Float64[
        20.0,    # n
        100.0,   # v_max_D
        0.80,    # v_max_M
        0.40,    # v_max_M2
        0.025,   # v_max_I
        9e-2,    # v_max_C
        0.02,    # v_max_T
        0.02,    # v_max_E_ext
        1e-2,    # v_max_inhib
        100.0,   # K_MD
        50.0,    # K_ID
        35.0,    # K_DM
        5.0,     # K_IM
        1.0,     # K_EeM
        1.0,     # K_EiM
        130.0,   # K_MI
        50.0,    # K_MC
        50.0,    # K_MT
        50.0,    # K_MEe
        10.0,    # K_IC
        1.5,     # k_transl
        0.23,    # k_deg_mRNA
        0.1,     # k_deg_D
        4e-4,    # k_deg_M
        1e-3,    # k_deg_I
        4e-2,    # k_deg_C
        0.02,    # k_deg_E_ext
        0.02,    # k_deg_E_int
        3.0,     # v_max_diff_D
        5.0,     # v_max_diff_E_ext
        0.1,     # v_max_diff_C
        5.0,     # K_D_diff
        0.1,     # k_echo_int
    ]

    const PARAM_LO = Float64[
        12.0,    # n              = 0.6 × 20.0
        60.0,    # v_max_D        = 0.6 × 100.0
        0.48,    # v_max_M        = 0.6 × 0.80
        0.24,    # v_max_M2       = 0.6 × 0.40
        0.015,   # v_max_I        = 0.6 × 0.025
        5.4e-2,  # v_max_C        = 0.6 × 0.09
        0.012,   # v_max_T        = 0.6 × 0.02
        0.012,   # v_max_E_ext    = 0.6 × 0.02
        6e-3,    # v_max_inhib    = 0.6 × 0.01
        60.0,    # K_MD           = 0.6 × 100.0
        30.0,    # K_ID           = 0.6 × 50.0
        21.0,    # K_DM           = 0.6 × 35.0
        3.0,     # K_IM           = 0.6 × 5.0
        0.6,     # K_EeM          = 0.6 × 1.0
        0.6,     # K_EiM          = 0.6 × 1.0
        78.0,    # K_MI           = 0.6 × 130.0
        30.0,    # K_MC           = 0.6 × 50.0
        30.0,    # K_MT           = 0.6 × 50.0
        30.0,    # K_MEe          = 0.6 × 50.0
        6.0,     # K_IC           = 0.6 × 10.0
        0.9,     # k_transl       = 0.6 × 1.5
        0.138,   # k_deg_mRNA     = 0.6 × 0.23
        0.06,    # k_deg_D        = 0.6 × 0.1
        2.4e-4,  # k_deg_M        = 0.6 × 4e-4
        6e-4,    # k_deg_I        = 0.6 × 1e-3
        0.024,   # k_deg_C        = 0.6 × 0.04
        0.012,   # k_deg_E_ext    = 0.6 × 0.02
        0.012,   # k_deg_E_int    = 0.6 × 0.02
        1.8,     # v_max_diff_D   = 0.6 × 3.0
        3.0,     # v_max_diff_E_ext = 0.6 × 5.0
        0.06,    # v_max_diff_C   = 0.6 × 0.1
        3.0,     # K_D_diff       = 0.6 × 5.0
        0.06,    # k_echo_int     = 0.6 × 0.1
    ]

    const PARAM_HI = Float64[
        30.0,    # n              = 1.5 × 20.0
        150.0,   # v_max_D        = 1.5 × 100.0
        1.2,     # v_max_M        = 1.5 × 0.80
        0.6,     # v_max_M2       = 1.5 × 0.40
        0.0375,  # v_max_I        = 1.5 × 0.025
        0.135,   # v_max_C        = 1.5 × 0.09
        0.03,    # v_max_T        = 1.5 × 0.02
        0.03,    # v_max_E_ext    = 1.5 × 0.02
        0.015,   # v_max_inhib    = 1.5 × 0.01
        150.0,   # K_MD           = 1.5 × 100.0
        75.0,    # K_ID           = 1.5 × 50.0
        52.5,    # K_DM           = 1.5 × 35.0
        7.5,     # K_IM           = 1.5 × 5.0
        1.5,     # K_EeM          = 1.5 × 1.0
        1.5,     # K_EiM          = 1.5 × 1.0
        195.0,   # K_MI           = 1.5 × 130.0
        75.0,    # K_MC           = 1.5 × 50.0
        75.0,    # K_MT           = 1.5 × 50.0
        75.0,    # K_MEe          = 1.5 × 50.0
        15.0,    # K_IC           = 1.5 × 10.0
        2.25,    # k_transl       = 1.5 × 1.5
        0.345,   # k_deg_mRNA     = 1.5 × 0.23
        0.15,    # k_deg_D        = 1.5 × 0.1
        6e-4,    # k_deg_M        = 1.5 × 4e-4
        1.5e-3,  # k_deg_I        = 1.5 × 1e-3
        0.06,    # k_deg_C        = 1.5 × 0.04
        0.03,    # k_deg_E_ext    = 1.5 × 0.02
        0.03,    # k_deg_E_int    = 1.5 × 0.02
        4.5,     # v_max_diff_D   = 1.5 × 3.0
        7.5,     # v_max_diff_E_ext = 1.5 × 5.0
        0.15,    # v_max_diff_C   = 1.5 × 0.1
        7.5,     # K_D_diff       = 1.5 × 5.0
        0.15,    # k_echo_int     = 1.5 × 0.1
    ]

    # Sigma par paramètre en espace log
    # exp(±σ) donne les bornes multiplicatives depuis θ₀ :
    #   σ = 1.10 → ×0.33 à ×3.0   (standard)
    #   σ = 0.80 → ×0.45 à ×2.2   (serré : params critiques pour les timescales)
    #   σ = 1.35 → ×0.26 à ×3.9   (large : params secondaires)
    const SIGMA_VEC = Float64[
        0.30,   # n              — switch Hill, critique
        0.30,   # v_max_D        — gain D
        0.30,   # v_max_M        — intégrateur D→M
        0.30,   # v_max_M2       — Hebbian, explorable
        0.30,   # v_max_I        — inhibition
        0.30,   # v_max_C        — consolidation
        0.30,   # v_max_T        — trace
        0.30,   # v_max_E_ext    — écho externe
        0.30,   # v_max_inhib    — inhibition C×M
        0.30,   # K_MD           — seuil spike, critique
        0.30,   # K_ID           — seuil inhibition D, critique
        0.30,   # K_DM           — seuil intégration, critique
        0.30,   # K_IM
        0.30,   # K_EeM
        0.30,   # K_EiM
        0.30,   # K_MI           — seuil activation I, critique
        0.30,   # K_MC
        0.30,   # K_MT
        0.30,   # K_MEe
        0.30,   # K_IC
        0.30,   # k_transl
        0.30,   # k_deg_mRNA
        0.30,   # k_deg_D        — timescale spike, critique
        0.30,   # k_deg_M        — timescale mémoire
        0.30,   # k_deg_I        — timescale inhibition
        0.30,   # k_deg_C
        0.30,   # k_deg_E_ext
        0.30,   # k_deg_E_int
        0.30,   # v_max_diff_D
        0.30,   # v_max_diff_E_ext
        0.30,   # v_max_diff_C
        0.30,   # K_D_diff
        0.30,   # k_echo_int
    ]
end

# ==============================================================================
# 3. ÉVALUATION + TRACKING THREAD-SAFE
# ==============================================================================

# Lock + état global mis à jour directement dans evaluate_candidate
# (state.value reste Inf avec :thread dans Evolutionary.jl — bug connu)
const eval_lock        = ReentrantLock()
const eval_count       = Ref(0)
const best_loss_global = Ref(Inf)
const best_θ_global    = Ref(zeros(length(THETA0)))

# CSV de mappage : une ligne par évaluation
# colonnes : eval, restart, loss, θ₁, ..., θ₃₂
const MAP_CSV = Ref("")   # initialisé après run_id

function evaluate_candidate(φ_norm::Vector{Float64})::Float64
    # φ_norm est normalisé : φ_norm = φ / SIGMA_VEC  (φ = log(θ))
    # → on dénormalise, puis on décode
    φ = φ_norm .* SIGMA_VEC
    θ_c = clamp.(exp.(φ), PARAM_LO, PARAM_HI)
    loss = BioSim.run_hebbian_simulation(θ_c, GATE_NAME, CFG)

    lock(eval_lock) do
        eval_count[] += 1
        n = eval_count[]

        # Mise à jour du meilleur global
        if isfinite(loss) && loss < best_loss_global[]
            best_loss_global[] = loss
            best_θ_global[]    = copy(θ_c)
            @printf "  [NEW BEST] eval %d | loss = %.4f\n" n loss
            flush(stdout)
        end

        # Log CSV (toujours, même si Inf)
        open(MAP_CSV[], "a") do f
            print(f, "$n,$(isfinite(loss) ? loss : "Inf")")
            for v in θ_c; print(f, ",$v"); end
            println(f)
        end
    end

    return loss
end

# ==============================================================================
# 4. CMA-ES AVEC RESTARTS
# ==============================================================================
using Evolutionary

θ₀ = THETA0
φ₀_norm = log.(θ₀) ./ SIGMA_VEC   # point de départ normalisé

mkpath(OUTPUT_DIR)
run_id = Dates.format(now(), "yyyymmdd_HHMMSS")

# Initialisation du CSV de mappage
MAP_CSV[] = joinpath(OUTPUT_DIR, "landscape_$run_id.csv")
open(MAP_CSV[], "w") do f
    header = "eval,loss," * join(["theta_$i" for i in 1:length(θ₀)], ",")
    println(f, header)
end
println("  landscape CSV → $(MAP_CSV[])")

# --- Hyperparamètres ---
# sigma0 = 0.3 en espace log → exp(±0.3) ≈ ×0.74 / ×1.35 (±30% multiplicatif)
# Cohérent pour tous les paramètres quelle que soit leur échelle (1e-5 à 500)
const SIGMA0            = 0.3
const LAMBDA            = 4 + floor(Int, 3 * log(length(θ₀)))  # ≈ 14
const GENS_PER_RESTART  = 30    # générations avant de décider de repartir
const N_RESTARTS        = 10    # max restarts (= 300 générations d'exploration total)
const EXPLOIT_GENS      = 270   # générations supplémentaires si signal trouvé
const SIGNAL_THRESHOLD  = 0.65  # en dessous de log(2)≈0.693 : signal réel détecté
const CHECKPOINT_EVERY  = 10

println("\nCMA-ES Hebbian — exploration par restarts aléatoires")
println("  gate            = $GATE_NAME")
println("  params          = $(length(θ₀))")
println("  threads         = $(Threads.nthreads())")
println("  sigma0          = $SIGMA0")
println("  lambda          = $LAMBDA")
println("  gens/restart    = $GENS_PER_RESTART  (max $N_RESTARTS restarts)")
println("  signal_threshold= $SIGNAL_THRESHOLD")
println("  output          = $OUTPUT_DIR\n")

gen_counter = Ref(0)   # compteur global de générations

function make_callback(restart_id)
    local_gen = Ref(0)

    return function(state)
        # state.value est toujours Inf avec :thread → on lit best_loss_global[]
        local_gen[] += 1
        gen_counter[] += 1
        gen   = local_gen[]
        total = gen_counter[]

        @printf "Restart %2d | Gen %3d (total %4d) | evals = %4d | best = %.4f\n" restart_id gen total eval_count[] best_loss_global[]
        flush(stdout)

        if total % CHECKPOINT_EVERY == 0
            path = joinpath(OUTPUT_DIR, "checkpoint_$run_id.jld2")
            save_object(path, Dict(
                "theta_best" => best_θ_global[],
                "loss_best"  => best_loss_global[],
                "eval_count" => eval_count[],
                "total_gen"  => total,
                "restart"    => restart_id,
                "date"       => string(now()),
            ))
            @printf "  [Checkpoint gen %d] → %s\n" total basename(path)
            flush(stdout)
        end

        return true
    end
end

# ==============================================================================
# 5. BOUCLE PRINCIPALE
# ==============================================================================
global signal_found = false

for restart in 1:N_RESTARTS

    # Choix du point de départ (en espace log)
    φ_start = if restart == 1
        copy(φ₀_norm)   # premier restart : θ₀ validé, normalisé
    else
        # tirage aléatoire dans [log(PARAM_LO), log(PARAM_HI)], normalisé
        (log.(PARAM_LO) .+ rand(length(PARAM_LO)) .* (log.(PARAM_HI) .- log.(PARAM_LO))) ./ SIGMA_VEC
    end

    println("\n" * "="^60)
    @printf "RESTART %d / %d — θ %s\n" restart N_RESTARTS (restart == 1 ? "par défaut" : "aléatoire")
    println("="^60)

    cb = make_callback(restart)

    n_gens = signal_found ? EXPLOIT_GENS : GENS_PER_RESTART

    Evolutionary.optimize(
        evaluate_candidate,
        φ_start,
        CMAES(sigma0 = 1.0, lambda = LAMBDA),  # sigma absorbé dans SIGMA_VEC
        Evolutionary.Options(
            iterations      = n_gens * LAMBDA,
            abstol          = 0.0,
            reltol          = 0.0,
            store_trace     = true,
            parallelization = :thread,
            callback        = cb,
        )
    )

    @printf "\n  → Meilleur global après restart %d : %.4f\n" restart best_loss_global[]

    # Signal détecté → mode exploitation (plus de restarts)
    if best_loss_global[] < SIGNAL_THRESHOLD && !signal_found
        global signal_found = true
        println("\n  *** Signal trouvé (loss < $SIGNAL_THRESHOLD) — bascule en exploitation ***")
    elseif signal_found
        break  # exploitation terminée
    else
        @printf "  → Stagnation (%.4f ≥ %.4f) — nouveau tirage aléatoire\n" best_loss_global[] SIGNAL_THRESHOLD
    end
end

# ==============================================================================
# 6. SAUVEGARDE FINALE
# ==============================================================================
circuit_ref, _ = BioKan.create_hebbian_model(:node_generic)
param_names    = string.(parameters(circuit_ref))
best_params_named = Dict(param_names[i] => best_θ_global[][i] for i in eachindex(best_θ_global[]))

@printf "\nTerminé — Loss finale : %.4f | Évaluations totales : %d | Générations : %d\n" best_loss_global[] eval_count[] gen_counter[]

save_object(joinpath(OUTPUT_DIR, "best_params_$run_id.jld2"), Dict(
    "theta_best"        => best_θ_global[],
    "loss_best"         => best_loss_global[],
    "best_params_named" => best_params_named,
    "theta_init"        => θ₀,
    "param_bounds_lo"   => PARAM_LO,
    "param_bounds_hi"   => PARAM_HI,
    "gate_name"         => string(GATE_NAME),
    "n_bacteries"       => CFG.n_bacteries,
    "total_evals"       => eval_count[],
    "total_gens"        => gen_counter[],
    "landscape_csv"     => MAP_CSV[],
    "date"              => string(now()),
))

println("Résultat : $(OUTPUT_DIR)/best_params_$run_id.jld2")
println("\nMeilleurs paramètres :")
for (name, val) in sort(collect(best_params_named))
    @printf "   %-20s = %.4g\n" name val
end
