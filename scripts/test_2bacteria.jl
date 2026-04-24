# ==============================================================================
# test_2bacteria.jl — Test mécanisme B1 → B2 (2 bactéries, pas de forcing output)
#
# B1 reçoit des spikes de D injectés manuellement.
# B2 est libre — on observe si le signal se propage.
#
# Usage :
#   julia --project=. scripts/test_2bacteria.jl
# ==============================================================================

using Pkg, Printf, Dates
PROJECT_ROOT = dirname(@__DIR__)
println("source du directory : $PROJECT_ROOT")
Pkg.activate(PROJECT_ROOT)

using Catalyst, JumpProcesses, OrdinaryDiffEq, Statistics, Random, LinearAlgebra
using Plots

include(joinpath(PROJECT_ROOT, "src", "BioKan.jl"))
using .BioKan

println("=== Test 2 bactéries ===\n")

# ==============================================================================
# 1. PARAMÈTRES (identiques à test_pipeline)
# ==============================================================================
θ₀ = Float64[
    20.0,     # n
    100.0,     # v_max_D    ← ÉTEINT : on ne veut pas M→D pour ce test
    0.5,     # v_max_M    ← SEUL ACTIF : D→M (intégration pure)
    0.1,  #v_max_M2
    0.025,     # v_max_I
    9e-2,     # v_max_C 1
    0.02,     # v_max_T
    0.02,     # v_max_E_ext
    1e-2,     # v_max_inhib

    100.0,    # K_MD
    50.0,     # K_ID
    15.0,     # K_DM. 40
    5.0,    # K_IM
    1.0,    # K_EeM
    1.0,    # K_EiM
    130.0,   # K_MI
    50.0,   # K_MC
    50.0,    # K_MT
    50.0,    # K_MEe
    10.0,    # K_IC

    1.5,     # k_transl

    0.23,     # k_deg_mRNA
    0.1,     # k_deg_D
    4e-4,     # k_deg_M
    1e-3,     # k_deg_I
    4e-2,     # k_deg_C 0.01 ou jsp quoi 
    0.02,     # k_deg_E_ext
    0.02,    # k_deg_E_int

    3.0,     # v_max_diff_D
    5.0,     # v_max_diff_E_ext
    0.1,  # v_max_diff_C
    5.0,     # K_D_diff

    0.1,     # k_echo_int
]

# ==============================================================================
# 2. CONFIG RÉSEAU ET DIFFUSION
# ==============================================================================
# Distance inter-bactéries = 1e-5 m (espacement grille cubique : d_max/n_segments = 3e-5/3)
d_max    = 3e-5
spacing  = d_max / 3       # = 1e-5 m  (même que dans test_pipeline)
R_cell   = 0.5e-6
dt       = 1/10
T_total  = 20000.0
total_steps = Int(T_total / dt)
time_vec = (1:total_steps) .* dt

species_names = [
    :D, :M, :I, :C, :E_int, :T, :E_ext,
    :D_diff, :C_diff, :E_ext_diff,
    :mRNA_D, :mRNA_M, :mRNA_I, :mRNA_C, :mRNA_T, :mRNA_E_ext
]
n_species = length(species_names)

D_dict = Dict{Symbol, Float64}(
    :D => 0.0, :M => 0.0, :I => 0.0, :C => 0.0,
    :E_int => 0.0, :T => 0.0, :E_ext => 0.0,
    :D_diff => 1.0e-9, :C_diff => 5.0e-9, :E_ext_diff => 1.0e-9,
    :mRNA_D => 0.0, :mRNA_M => 0.0, :mRNA_I => 0.0,
    :mRNA_C => 0.0, :mRNA_T => 0.0, :mRNA_E_ext => 0.0
)
gamma_dict = Dict{Symbol, Float64}(
    :D => 0.0, :M => 0.0, :I => 0.0, :C => 0.0,
    :E_int => 0.0, :T => 0.0, :E_ext => 0.0,
    :D_diff => 2.9e-2, :C_diff => 7e-2, :E_ext_diff => 2.9e-2,
    :mRNA_D => 0.0, :mRNA_M => 0.0, :mRNA_I => 0.0,
    :mRNA_C => 0.0, :mRNA_T => 0.0, :mRNA_E_ext => 0.0
)
diffusion_targets = Dict{Symbol, Symbol}(
    :D_diff => :D, :C_diff => :C, :E_ext_diff => :E_ext,
    # les non-diffusibles pointent sur eux-mêmes (jamais utilisés)
    :D => :D, :M => :M, :I => :I, :C => :C,
    :E_int => :E_int, :T => :T, :E_ext => :E_ext,
    :mRNA_D => :mRNA_D, :mRNA_M => :mRNA_M, :mRNA_I => :mRNA_I,
    :mRNA_C => :mRNA_C, :mRNA_T => :mRNA_T, :mRNA_E_ext => :mRNA_E_ext
)

# ==============================================================================
# 3. CONSTRUCTION DU RÉSEAU (2 bactéries)
# ==============================================================================
circuit, _ = BioKan.create_hebbian_model(:node)


params_dict = Dict(parameters(circuit) .=> θ₀)
u0_raw      = Dict(s => 0.0 for s in species_names)
u0_dict     = BioKan.map_symbols_to_species(circuit, u0_raw)

net = BioKan.BioNetwork(d_max, 3)

b1 = BioKan.Bacterium(1, [0.0,     0.0], circuit, params_dict, u0_dict; mode=:ssa)
b2 = BioKan.Bacterium(2, [spacing, 0.0], circuit, params_dict, u0_dict; mode=:ssa)

BioKan.add_bacterium!(net, b1)
BioKan.add_bacterium!(net, b2)
BioKan.build_edges!(net)

@printf "Distance B1→B2 : %.2e m  (d_max = %.2e m)\n" spacing d_max

weights = BioKan.compute_static_coupling_physics(
    net.edges, D_dict, gamma_dict, species_names, R_cell, dt)

# ==============================================================================
# 4. PROTOCOLE D'INJECTION (spikes de D dans B1 uniquement)
# Deux modes disponibles : :burst ou :poisson
# ==============================================================================
input_mode    = :poisson   # ← changer ici pour switcher de protocole :burst ou :poisson
max_density   = 60.0       # amplitude d'un spike (molécules par step)

# --- Protocole BURST : 4 spikes groupés, puis silence ---
stimulus_duration = 2.0    # durée d'un spike (s)
inter_spike       = 500    # silence entre les spikes du burst (s)
silence           = 10000  # silence entre deux bursts (s)
t_start_stim      = 5.0    # début du premier burst

burst_period  = 4 * (stimulus_duration + inter_spike) + silence
spike_offsets = [k * (stimulus_duration + inter_spike) for k in 0:3]

burst_times = Float64[]
let t_burst = t_start_stim
    while t_burst + last(spike_offsets) + stimulus_duration < T_total
        for offset in spike_offsets
            push!(burst_times, t_burst + offset)
        end
        t_burst += burst_period
    end
end

# --- Protocole POISSON : spikes aléatoires, taux moyen = 1/s ---
poisson_rate      = 1/180.0  # 1 spike par minute en moyenne
spike_duration_p  = 2.0     # durée d'un spike (s)

poisson_times = Float64[]
let t_p = t_start_stim
    while t_p < T_total
        isi = randexp() / poisson_rate   # inter-spike interval exponentiel
        t_p += isi
        t_p < T_total && push!(poisson_times, t_p)
    end
end

# --- Sélection du protocole actif ---
spike_times = input_mode == :poisson ? poisson_times : burst_times
spike_dur   = input_mode == :poisson ? spike_duration_p : stimulus_duration

println("Protocole : $input_mode — $(length(spike_times)) spikes")

Input_B1 = zeros(total_steps)
for t_spike in spike_times
    i_start = round(Int, t_spike / dt) + 1
    i_end   = round(Int, (t_spike + spike_dur) / dt)
    i_start = clamp(i_start, 1, total_steps)
    i_end   = clamp(i_end,   1, total_steps)
    Input_B1[i_start:i_end] .= max_density
end

# Pour les marqueurs visuels on marque le début de chaque burst
stim_vlines = [t_start_stim + k * burst_period for k in 0:floor(Int, (T_total - t_start_stim) / burst_period) - 1]
println("Protocole : $(length(spike_times)) spikes ($(length(stim_vlines)) bursts × 3), cycle=$(burst_period)s")


# ==============================================================================
# 5. BOUCLE DE SIMULATION
# ==============================================================================
hist_B1 = zeros(Float32, total_steps, n_species)
hist_B2 = zeros(Float32, total_steps, n_species)

diffusible = [D_dict[species_names[s]] > 1e-40        for s in 1:n_species]
decay_vec  = [exp(-gamma_dict[species_names[s]] * dt) for s in 1:n_species]
target_vec = [diffusion_targets[species_names[s]]     for s in 1:n_species]

flux_emissions = zeros(Float64, 2, n_species)

t_start = time()
for step in 1:total_steps

    # A. INJECTION dans B1
    if Input_B1[step] > 0.0
        BioKan.add_input!(net.nodes[1], :D, Input_B1[step])
    end

    # B. BIOLOGIE INTERNE
    BioKan.step_bacterium!(net.nodes[1], dt)
    BioKan.step_bacterium!(net.nodes[2], dt)

    # C. FLUX DIFFUSIBLES
    fill!(flux_emissions, 0.0)
    for (i, b) in [(1, net.nodes[1]), (2, net.nodes[2])]
        for s in 1:n_species
            diffusible[s] || continue
            v = BioKan.get_species(b, species_names[s])
            v > 0.0 && (flux_emissions[i, s] = v)
        end
    end

    # D. TRANSPORT
    received = BioKan.propagate_signals_instantaneous!(
        weights, flux_emissions, 2, n_species)

    # E. UPDATE
    for (i, b) in [(1, net.nodes[1]), (2, net.nodes[2])]
        for s in 1:n_species
            if diffusible[s]
                amt = max(0.0, received[i, s] * decay_vec[s])
                BioKan.set_species!(b, target_vec[s],
                    BioKan.get_species(b, target_vec[s]) + amt)
                BioKan.set_species!(b, species_names[s], 0.0)
            end
        end
    end

    # F. ENREGISTREMENT
    for s in 1:n_species
        hist_B1[step, s] = Float32(max(0.0, BioKan.get_species(net.nodes[1], species_names[s])))
        hist_B2[step, s] = Float32(max(0.0, BioKan.get_species(net.nodes[2], species_names[s])))
    end
end
@printf "Simulation terminée en %.1f s\n" (time() - t_start)


# ==============================================================================
# 6. DIAGNOSTICS
# ==============================================================================
println("\n--- Max par espèce ---")
for (s, name) in enumerate(species_names)
    m1 = maximum(hist_B1[:, s])
    m2 = maximum(hist_B2[:, s])
    (m1 > 0.01 || m2 > 0.01) && @printf "  %-12s B1=%.3f   B2=%.3f\n" name m1 m2
end

# ==============================================================================
# 7. TRACÉ
# ==============================================================================
t = collect(time_vec)

idx = Dict(name => findfirst(==(name), species_names) for name in species_names)

# Debug : évolution de D, M, I, C dans B1 autour du 2ème burst
t2 = spike_times[1]   # début du 2ème burst (spike 2)
i_start_dbg = max(1, round(Int, (t2 - 50.0) / dt))
i_end_dbg   = min(total_steps, round(Int, (t2 + 200.0) / dt))


# Marqueurs de stimulus (lignes verticales)
# stim_vlines déjà défini dans le bloc protocole ci-dessus

function make_panel(hist, title_str)
    p = plot(title=title_str, xlabel="Temps (s)", ylabel="mol",
             legend=:outertopright, lw=1.2)
    # Pas de vlines sur la vue complète (trop denses avec 1000+ bursts)
    plot!(p, t, Float64.(hist[:, idx[:D]]),     label="D",     color=:steelblue)
    plot!(p, t, Float64.(hist[:, idx[:M]]),     label="M",     color=:crimson)
    plot!(p, t, Float64.(hist[:, idx[:I]]),     label="I",     color=:green)
    plot!(p, t, Float64.(hist[:, idx[:E_int]]), label="E_int", color=:brown)
    plot!(p, t, Float64.(hist[:, idx[:C]]),     label="C",     color=:purple)
    return p
end
println("C(t=1s)  B1=", hist_B1[10, idx[:C]], "  B2=", hist_B2[10, idx[:C]])
println("C(t=10s) B1=", hist_B1[100, idx[:C]], "  B2=", hist_B2[100, idx[:C]])
println("C(t=50s) B1=", hist_B1[500, idx[:C]], "  B2=", hist_B2[500, idx[:C]])

# Panel input — on trace l'enveloppe (trop de points pour tout afficher)
p_input = plot(t, Input_B1, color=:steelblue, lw=0.5, fill=0, alpha=0.4,
               title="Signal injecté (D dans B1)", xlabel="Temps (s)",
               ylabel="mol/step", label="D_injecté", legend=:outertopright)

p_B1 = make_panel(hist_B1, "B1 — Input (forcing D)")
p_B2 = make_panel(hist_B2, "B2 — Output (libre)")

# Zoom sur un spike : fenêtre autour du 1er stimulus
t_zoom_start = t_start_stim - 2.0
t_zoom_end   = t_start_stim + 30.0
zoom_mask    = t_zoom_start .<= t .<= t_zoom_end

zoom_vlines = filter(v -> t_zoom_start <= v <= t_zoom_end, stim_vlines)

function make_zoom(hist, title_str)
    p = plot(title=title_str, xlabel="Temps (s)", ylabel="mol",
             legend=:outertopright, lw=1.5)
    vline!(p, zoom_vlines, color=:black, lw=0.8, ls=:dot, label="burst", alpha=0.7)
    plot!(p, t[zoom_mask], Float64.(hist[zoom_mask, idx[:D]]),     label="D",     color=:steelblue)
    plot!(p, t[zoom_mask], Float64.(hist[zoom_mask, idx[:M]]),     label="M",     color=:crimson)
    plot!(p, t[zoom_mask], Float64.(hist[zoom_mask, idx[:I]]),     label="I",     color=:green)
    plot!(p, t[zoom_mask], Float64.(hist[zoom_mask, idx[:E_int]]), label="E_int", color=:brown)
    return p
end

p_zoom_B1 = make_zoom(hist_B1, "Zoom B1 — 1er stimulus")
p_zoom_B2 = make_zoom(hist_B2, "Zoom B2 — réponse propagée")

fig = plot(p_input, p_B1, p_B2, p_zoom_B1, p_zoom_B2,
           layout=(5, 1), size=(1200, 1600), left_margin=10Plots.mm)

out_path = joinpath(PROJECT_ROOT, "outputs",
                    "test_2bacteria_$(Dates.format(now(), "yyyymmdd_HHMMSS")).png")
mkpath(dirname(out_path))
savefig(fig, out_path)
println("\nPlot sauvegardé : $out_path")
