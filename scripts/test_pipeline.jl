# ==============================================================================
# test_pipeline.jl — Validation du pipeline Hebbian end-to-end (version threadée)
#
# Lance une simulation avec les paramètres par défaut, affiche la loss
# et trace les séries temporelles de D pour les 4 nœuds rôles + 1 interneurone.
#
# Usage :
#   julia --project=. --threads=auto scripts/test_pipeline.jl
# ==============================================================================

using Pkg, Printf, Dates
PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT)

using Catalyst, JumpProcesses, OrdinaryDiffEq, Statistics, Random, Distributions, LinearAlgebra
using Plots

include(joinpath(PROJECT_ROOT, "src",     "BioKan.jl"))
include(joinpath(PROJECT_ROOT, "scripts", "simulation.jl"))
using .BioKan
using .BioSim

println("=== Test pipeline Hebbian ($(Threads.nthreads()) threads) ===\n")

# --- 1. Paramètres --- (modifie ici pour explorer manuellement)
# Ordre : n, v_max_D, v_max_M, v_max_I, v_max_C, v_max_T, v_max_E_ext, v_max_inhib,
#         K_MD, K_ID, K_DM, K_IM, K_EeM, K_EiM, K_MI, K_MC, K_MT, K_MEe, K_IC,
#         k_transl, k_deg_mRNA, k_deg_D, k_deg_M, k_deg_I, k_deg_C, k_deg_E_ext, k_deg_E_int,
#         v_max_diff_D, v_max_diff_E_ext, v_max_diff_C, K_D_diff, k_echo_int
θ₀ = Float64[
    20.0,    # n
    100.0,   # v_max_D
    0.80,     # v_max_M    — voie D→M
    0.40,     # v_max_M2   — voie E_ext×E_int→M (Hebbian)
    0.025,   # v_max_I
    9e-2,    # v_max_C
    0.02,    # v_max_T
    0.02,    # v_max_E_ext
    1e-2,    # v_max_inhib
    100.0,   # K_MD
    50.0,    # K_ID
    15.0,    # K_DM
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
    0.001,    # k_deg_I 0.001
    4e-2,    # k_deg_C
    0.02,    # k_deg_E_ext
    0.02,    # k_deg_E_int
    3.0,     # v_max_diff_D
    5.0,     # v_max_diff_E_ext
    0.1,     # v_max_diff_C
    5.0,     # K_D_diff
    0.1,     # k_echo_int
]

cfg = BioSim.default_hebbian_config()

println("Réseau       : $(cfg.n_bacteries) bactéries")
println("Paramètres   : $(length(θ₀))")
println("Gate         : XOR\n")

# --- 2. Vérification de l'ordre des paramètres ---
_circuit_check, _ = BioKan.create_hebbian_model(:hebbian_node)
println("Ordre des paramètres utilisé :")
for (i, p) in enumerate(parameters(_circuit_check))
    println("  $i : $p = $(θ₀[i])")
end
println()

# --- 3. Simulation ---
println("Lancement de la simulation...")
t_start = time()

loss, history, time_steps, node_roles, Input_A, Input_B, Output_0, Output_1, mask, net =
    BioSim.run_hebbian_simulation_threaded(θ₀, :XOR, cfg)

t_elapsed = time() - t_start

# --- Diagnostic max par espèce ---
println("\n--- Max par espèce (tout le réseau, toute la simulation) ---")
for (s, name) in enumerate(cfg.species_names)
    println("  $name : max=$(maximum(history[:,:,s]))")
end

# --- 3. Résultat ---
println("\n--- Résultat ---")
if isinf(loss)
    println("[EXPLOSION] Concentrations aberrantes détectées — simulation arrêtée.")
    println("  → CMA-ES recevra loss = Inf pour ce candidat.")
else
    @printf "Loss        = %.4f  (réf. aléatoire = %.4f)\n" loss log(2)
    if loss < log(2)
        println("Le réseau fait mieux qu'un classifieur aléatoire.")
    else
        println("Le réseau ne fait pas mieux qu'un classifieur aléatoire (attendu avec θ par défaut).")
    end
end
@printf "Durée       = %.1f s\n" t_elapsed

# --- 4. Tracés (même si explosion, on trace ce qui a été enregistré) ---
species_names = cfg.species_names
d_idx    = findfirst(==(:D), species_names)
K_D_diff = θ₀[31]
t        = collect(time_steps)

role_ids    = [node_roles[:Input_A], node_roles[:Input_B],
               node_roles[:Output_0], node_roles[:Output_1]]
role_labels = ["Input_A ($(node_roles[:Input_A]))",
               "Input_B ($(node_roles[:Input_B]))",
               "Output_0 ($(node_roles[:Output_0]))",
               "Output_1 ($(node_roles[:Output_1]))"]

inter_pool = [id for id in 1:cfg.n_bacteries if id ∉ role_ids]
inter_ids  = inter_pool[randperm(length(inter_pool))[1:min(3, length(inter_pool))]]
all_ids    = vcat(role_ids, inter_ids)
all_labels = vcat(role_labels, ["Interneurone ($id)" for id in inter_ids])

# Indices des espèces à tracer
m_idx     = findfirst(==(:M),     species_names)
i_idx     = findfirst(==(:I),     species_names)
c_idx     = findfirst(==(:C),     species_names)
t_idx     = findfirst(==(:T),     species_names)
e_int_idx = findfirst(==(:E_int), species_names)

# Fenêtres d'évaluation de la loss (phase test uniquement)
# vert  = expected Output_1 | orange = expected Output_0
function add_eval_windows!(p)
    for trial in mask
        col = trial.expected == 1 ? RGBA(0.0,0.6,0.0,0.15) : RGBA(1.0,0.5,0.0,0.15)
        vspan!(p, [trial.t_start, trial.t_end], color=col, label="")
    end
    # Légende une seule fois (vlines fantômes)
    vline!(p, [-1.0], color=RGBA(0.0,0.6,0.0,0.6), lw=2, ls=:solid, label="eval Out1")
    vline!(p, [-1.0], color=RGBA(1.0,0.5,0.0,0.6), lw=2, ls=:solid, label="eval Out0")
end

# Subplot pattern injecté
p_inputs = plot(t, Input_A, label="Input_A", color=:blue, lw=1.5,
                title="Pattern injecté", ylabel="Densité", xlabel="Temps (s)")
plot!(p_inputs, t, Input_B, label="Input_B", color=:orange, lw=1.5)
add_eval_windows!(p_inputs)

p_outputs = plot(t, Output_0, label="Output_0", color=:teal, lw=1.5,
                 title="Outputs injectés", ylabel="Densité", xlabel="Temps (s)")
plot!(p_outputs, t, Output_1, label="Output_1", color=:red, lw=1.5)
add_eval_windows!(p_outputs)

panels = [p_inputs, p_outputs]
for (id, lab) in zip(all_ids, all_labels)
    d_trace     = Float64.(history[:, id, d_idx])
    m_trace     = Float64.(history[:, id, m_idx])
    i_trace     = Float64.(history[:, id, i_idx])
    c_trace     = Float64.(history[:, id, c_idx])
    t_trace     = Float64.(history[:, id, t_idx])
    e_int_trace = Float64.(history[:, id, e_int_idx])
    p = plot(t, d_trace,     label="D",     color=:steelblue, lw=1,
             title=lab, ylabel="mol", xlabel="Temps (s)")
    plot!(p, t, m_trace,     label="M",     color=:crimson,   lw=1)
    plot!(p, t, i_trace,     label="I",     color=:green,     lw=1)
    plot!(p, t, c_trace,     label="C",     color=:purple,    lw=1)
    plot!(p, t, t_trace,     label="T",     color=:orange,    lw=1)
    plot!(p, t, e_int_trace, label="E_int", color=:brown,     lw=1)
    hline!(p, [K_D_diff], label="K_D_diff", color=:red, lw=1, ls=:dash)
    add_eval_windows!(p)
    push!(panels, p)
end

println("\n--- Max par espèce (sur tout le réseau, toute la simulation) ---")
for (s, name) in enumerate(species_names)
    println("  $name : max=$(maximum(history[:,:,s]))")
end

# Plot moyenne par espèce sur toute la simulation (moyennée sur tous les noeuds)
species_to_plot = [:D, :M, :I, :C, :T, :E_int, :E_ext]
p_means = plot(title="Moyenne réseau par espèce", ylabel="mol", xlabel="Temps (s)", legend=:outertopright)
colors_mean = [:steelblue, :crimson, :green, :purple, :orange, :brown, :pink]
for (sp, col) in zip(species_to_plot, colors_mean)
    idx = findfirst(==(sp), species_names)
    isnothing(idx) && continue
    trace = dropdims(mean(history[:, :, idx], dims=2), dims=2)
    plot!(p_means, t, Float64.(trace), label=string(sp), color=col, lw=1.5)
end
push!(panels, p_means)

# Réseau 3D
push!(panels, BioKan.plot_bionetwork_3d(net, node_roles))

fig = plot(panels..., layout=(length(panels), 1),
           size=(1200, 300 * length(panels)), legend=:outertopright)

out_path = joinpath(PROJECT_ROOT, "outputs", "test_pipeline_$(Dates.format(now(), "yyyymmdd_HHMMSS")).png")
mkpath(dirname(out_path))
savefig(fig, out_path)
println("\nPlot sauvegardé : $out_path")
