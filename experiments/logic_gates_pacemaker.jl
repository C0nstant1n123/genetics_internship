# logic_gates_pacemaker.jl — 2-input logic gates on the 8-bacteria topology,
# with Input_0 (B8) driven as a PACEMAKER: it spikes at a fixed period
# (pacemaker_period), independently of the protocol, forcing the spike directly
# rather than through an injected D_diff signal.
#
# The target gate (:XOR, :AND, :OR, …) is passed as ARG 1.
#
# Usage: julia --project=. experiments/logic_gates_pacemaker.jl [GATE] [BATCH_ID]

using Pkg, Printf, Dates
PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT)

using Catalyst, JumpProcesses, OrdinaryDiffEq, Statistics, Random, LinearAlgebra
using Plots

include(joinpath(PROJECT_ROOT, "src", "BioKan.jl"))
using .BioKan

println("=== Test XOR PACE — topologie custom (8 bactéries SSA) — lancé le $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS")) ===\n")
flush(stdout)

# 1. Model parameters (θ₀ index → name given inline)
r = 0.4
θ₀ = Float64[
    15.0,   # 1  n
    2.0,    # 2  m
    1.0,    # 3  l
    2.7,    # 4  v_max_M
    0.90,   # 5  v_max_learn
    0.16,   # 6  v_ground
    50e-4*r, # 7  v_max_CS
    0.7,    # 8  v_max_ES
    74.0,   # 9  K_DM
    50.0,   # 10 K_SM
    4.0,    # 11 K_CS
    20.0,   # 12 K_EiS
    1.0,    # 13 K_EeS
    200.0,  # 14 K_act_S
    15.0,  # 15 K_inh_S
    1.0,    # 16 K_IS
    40.0,   # 17 K_DS
    120.0,  # 18 K_spike
    2000.0, # 19 K_spike_O
    0.0,    # 20 v_max_spike_D
    0.0,    # 21 v_max_spike_E
    0.0,    # 22 v_max_spike_C
    0.0,    # 23 v_max_spike_T
    0.0,    # 24 v_max_spike_O
    0.15,   # 25 k_transl_M
    0.08*r, # 26 k_transl_S
    0.23,   # 27 k_deg_mRNA_M
    0.23,   # 28 k_deg_mRNA_S
    0.005,  # 29 k_deg_M
    0.001,  # 30 k_deg_E_int
    0.05,   # 31 k_deg_E_ext  (réduit pour dt=10 : 1.3 → exp(-13)≈0 détruisait E_ext en transit)
    0.0019, # 32 k_deg_C  (réduit proportionnellement à E_ext : 0.05/26 ≈ 0.0019)
    0.0,    # 33 k_deg_T
    0.2e-15*r,  # 34 k_deg_S
    0.02,   # 35 k_deg_D
    0.02,   # 36 k_deg_O
    0.1,    # 37 k_echo_int
    1.0,    # 38 v_max_diff_D
    0.5,    # 39 v_max_diff_E
    0.4,    # 40 v_max_diff_C
    0.01,   # 41 v_max_diff_O
    100.0,  # 42 K_D_diff
    1.0,    # 43 K_OD
]

# 2. CONFIG RÉSEAU ET DIFFUSION
spacing = 1e-5
d_max   = 1.2e-5
R_cell  = 0.5e-6
dt      = 10.0

species_names = [
    :D, :M, :I, :E_int, :T, :E_ext, :C, :O, :S,
    :D_diff, :E_ext_diff, :C_diff, :O_diff,
    :mRNA_M, :mRNA_S
]
n_species = length(species_names)

gamma_D     = θ₀[35]   # k_deg_D
gamma_C     = θ₀[32]   # k_deg_C
gamma_E_ext = θ₀[31]   # k_deg_E_ext
gamma_O     = θ₀[36]   # k_deg_O

lambda   = 0.15 * spacing   # D, E_ext
lambda_C = 0.15 * spacing   # C, O 0.5

D_dict = Dict{Symbol, Float64}(
    :D => 0.0, :M => 0.0, :I => 0.0, :E_int => 0.0, :T => 0.0,
    :E_ext => 0.0, :C => 0.0, :O => 0.0, :S => 0.0,
    :D_diff     => lambda^2   * gamma_D,
    :E_ext_diff => lambda^2   * gamma_E_ext,
    :C_diff     => lambda_C^2 * gamma_C,
    :O_diff     => lambda_C^2 * gamma_O,
    :mRNA_M => 0.0, :mRNA_S => 0.0
)
gamma_dict = Dict{Symbol, Float64}(
    :D => 0.0, :M => 0.0, :I => 0.0, :E_int => 0.0, :T => 0.0,
    :E_ext => 0.0, :C => 0.0, :O => 0.0, :S => 0.0,
    :D_diff     => gamma_D,
    :E_ext_diff => gamma_E_ext,
    :C_diff     => gamma_C,
    :O_diff     => gamma_O,
    :mRNA_M => 0.0, :mRNA_S => 0.0
)
diffusion_targets = Dict{Symbol, Symbol}(
    :D => :D, :M => :M, :I => :I, :E_int => :E_int, :T => :T,
    :E_ext => :E_ext, :C => :C, :O => :O, :S => :S,
    :D_diff => :D, :E_ext_diff => :E_ext, :C_diff => :C, :O_diff => :O,
    :mRNA_M => :mRNA_M, :mRNA_S => :mRNA_S
)

# 3. CONSTRUCTION DU RÉSEAU
#
# TODO : modifier les positions selon la topologie souhaitée
#        (s = spacing = 1e-5 m, d_max = 1.2e-5 m → connexion si dist ≤ d_max)
#
#=
s = spacing
h = s / sqrt(2)

circuit, _ = BioKan.create_hebbian_non_spike_model(:node)
params_dict = Dict(parameters(circuit) .=> θ₀)
u0_raw      = Dict(sp => 0.0 for sp in species_names)
u0_dict     = BioKan.map_symbols_to_species(circuit, u0_raw)

net = BioKan.BioNetwork(d_max, 10)


b1 = BioKan.Bacterium(1, [ 0.0,   0.0,   0.0], circuit, params_dict, u0_dict; mode=:ssa) # 
b2 = BioKan.Bacterium(2, [   s,   0.0,   0.0], circuit, params_dict, u0_dict; mode=:ssa) # INput A
b3 = BioKan.Bacterium(3, [  s,   s,   0.0], circuit, params_dict, u0_dict; mode=:ssa) # 
b4 = BioKan.Bacterium(4, [ 0.0,   s,   0.0], circuit, params_dict, u0_dict; mode=:ssa) # INput B
b5 = BioKan.Bacterium(5, [ 0.0,  0.0,   s], circuit, params_dict, u0_dict; mode=:ssa) # 
b6 = BioKan.Bacterium(6, [ 0.0,  s,   s], circuit, params_dict, u0_dict; mode=:ssa) # 
b7 = BioKan.Bacterium(7, [  0.0,  -s,  0.0], circuit, params_dict, u0_dict; mode=:ssa) # Pacemaker
b8 = BioKan.Bacterium(8, [ 0.5*s, s + h, 0.5*s], circuit, params_dict, u0_dict; mode=:ssa) # output
b9 = BioKan.Bacterium(9, [ 2*s, 0.0, 0.0], circuit, params_dict, u0_dict; mode=:ssa) # 
b10 = BioKan.Bacterium(10, [ 0.0, 0.0, 2*s], circuit, params_dict, u0_dict; mode=:ssa) # Input_B
for b in [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]
    BioKan.add_bacterium!(net, b)
end
BioKan.build_edges!(net)

n_bac = length(net.nodes)
=#
s = spacing

circuit, _ = BioKan.create_hebbian_non_spike_model(:node)
params_dict = Dict(parameters(circuit) .=> θ₀)
u0_raw      = Dict(sp => 0.0 for sp in species_names)
u0_dict     = BioKan.map_symbols_to_species(circuit, u0_raw)

# On garde d_max à 1.2s pour couper net toutes les diagonales à ~1.41s
net = BioKan.BioNetwork(d_max, 10)

# TOPOLOGIE 10 NŒUDS (version complète — décommenter pour revenir en arrière)
# Rôles : B1=relay_A, B2=relay_B, B3=relay_L, B4=relay_R, B5=hub,
#         B6=pre_out, B7=Output, B8=Pacemaker, B9=Input_A, B10=Input_B
# Connexions : B9→B1→B3, B10→B2→B4, B8↔B3↔B5↔B4, B3↔B6↔B4, B6→B7
#              Pacemaker B8 connecté à B5 (hub) à dist 1s
# Pour revenir à 10 nœuds : décommenter le bloc #= ... =# ci-dessous et commenter le bloc 9 nœuds

# TOPOLOGIE 10 NŒUDS — hub B5 séparé du pacemaker B8
# Rôles : B1=relay_A, B2=relay_B, B3=relay_L, B4=relay_R, B5=hub (S figé),
#         B6=pre_out, B7=Output, B8=Pacemaker, B9=Input_A, B10=Input_B
b8  = BioKan.Bacterium(8,  [ -1.7071*s,  1.7071*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa) # Pacemaker
b9  = BioKan.Bacterium(9,  [ -3.0*s,     0.0*s,     0.0], circuit, params_dict, u0_dict; mode=:ssa) # Input_A
b10 = BioKan.Bacterium(10, [  0.0*s,     3.0*s,     0.0], circuit, params_dict, u0_dict; mode=:ssa) # Input_B
b1  = BioKan.Bacterium(1,  [ -2.0*s,     0.0*s,     0.0], circuit, params_dict, u0_dict; mode=:ssa) # relay_A
b2  = BioKan.Bacterium(2,  [  0.0*s,     2.0*s,     0.0], circuit, params_dict, u0_dict; mode=:ssa) # relay_B
b5  = BioKan.Bacterium(5,  [ -1.0*s,     1.0*s,     0.0], circuit, params_dict, u0_dict; mode=:ssa) # hub
b3  = BioKan.Bacterium(3,  [ -1.0*s,     0.0*s,     0.0], circuit, params_dict, u0_dict; mode=:ssa) # relay_L
b4  = BioKan.Bacterium(4,  [  0.0*s,     1.0*s,     0.0], circuit, params_dict, u0_dict; mode=:ssa) # relay_R
b6  = BioKan.Bacterium(6,  [  0.0*s,     0.0*s,     0.0], circuit, params_dict, u0_dict; mode=:ssa) # pre_out
b7  = BioKan.Bacterium(7,  [  0.0*s,    -1.0*s,     0.0], circuit, params_dict, u0_dict; mode=:ssa) # Output
for b in [b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]; BioKan.add_bacterium!(net, b); end
BioKan.build_edges!(net); n_bac = length(net.nodes)
# IDs 1..10 consécutifs → bi = ID
relay_s_frozen = 100   # valeur de gel de B1/B2 si freeze_relays=true
hub_s_frozen   = 100  # valeur de gel du hub B5 (permanente)
BioKan.set_species!(net.nodes[1],  :S,  100.0)  # B1  relay_A
BioKan.set_species!(net.nodes[2],  :S,  100.0)  # B2  relay_B
BioKan.set_species!(net.nodes[3],  :S,  100.0)  # B3  relay_L
BioKan.set_species!(net.nodes[4],  :S,  100.0)  # B4  relay_R
BioKan.set_species!(net.nodes[5],  :S,  150.0)  # B5  hub (figé en permanence)
BioKan.set_species!(net.nodes[6],  :S,  170.0)  # B6  pre_out
BioKan.set_species!(net.nodes[7],  :S,  200.0)  # B7  Output
BioKan.set_species!(net.nodes[8],  :S,  200.0)  # B8  Pacemaker
BioKan.set_species!(net.nodes[9],  :S,  200.0)  # B9  Input_A
BioKan.set_species!(net.nodes[10], :S,  200.0)  # B10 Input_B

#= TOPOLOGIE 9 NŒUDS — Pacemaker (B8) remplace le hub (B5) à sa position
# Pour revenir à ce mode : commenter le bloc 10 nœuds ci-dessus et décommenter celui-ci
b9  = BioKan.Bacterium(9,  [ -3.0*s,  0.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa)
b10 = BioKan.Bacterium(10, [  0.0*s,  3.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa)
b1  = BioKan.Bacterium(1,  [ -2.0*s,  0.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa)
b2  = BioKan.Bacterium(2,  [  0.0*s,  2.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa)
b8  = BioKan.Bacterium(8,  [ -1.0*s,  1.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa)
b3  = BioKan.Bacterium(3,  [ -1.0*s,  0.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa)
b4  = BioKan.Bacterium(4,  [  0.0*s,  1.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa)
b6  = BioKan.Bacterium(6,  [  0.0*s,  0.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa)
b7  = BioKan.Bacterium(7,  [  0.0*s, -1.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa)
for b in [b1,b2,b3,b4,b8,b6,b7,b9,b10]; BioKan.add_bacterium!(net, b); end
BioKan.build_edges!(net); n_bac = length(net.nodes)
BioKan.set_species!(net.nodes[1], :S, 0.0); BioKan.set_species!(net.nodes[2], :S, 205.0)
BioKan.set_species!(net.nodes[3], :S, 542.0); BioKan.set_species!(net.nodes[4], :S, 1167.0)
BioKan.set_species!(net.nodes[8], :S, 1233.0); BioKan.set_species!(net.nodes[6], :S, 266.0)
BioKan.set_species!(net.nodes[7], :S, 0.0); BioKan.set_species!(net.nodes[9], :S, 9.0)
BioKan.set_species!(net.nodes[10], :S, 0.0)
=#

#= TOPOLOGIE 7 NŒUDS (version simplifiée — relay_A, relay_B, hub supprimés)
# Rôles : B1=Input_A, B2=Input_B, B3=relay_L, B4=relay_R,
#         B5=Pacemaker, B6=pre_out, B7=Output
# Connexions : B1→B3→B6→B7, B2→B4→B6, B5↔B3, B5↔B4
# (B5 occupe l'ancienne position du hub pour rester connecté à B3 et B4)
# Pour revenir à 10 nœuds : commenter ce bloc et décommenter le bloc 10 nœuds
# --- COUCHE 1 : Entrées (sur l'ancienne position des relais) ---
b1  = BioKan.Bacterium(1,  [ -2.0*s,    0.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa) # Input_A  (dist 1s à B3)
b2  = BioKan.Bacterium(2,  [  0.0*s,    2.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa) # Input_B  (dist 1s à B4)

# --- COUCHE 2 : Couche cachée ---
b3  = BioKan.Bacterium(3,  [ -1.0*s,    0.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa) # relay_L  (dist 1s à B1, B5, B6)
b4  = BioKan.Bacterium(4,  [  0.0*s,    1.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa) # relay_R  (dist 1s à B2, B5, B6)

# --- Pacemaker (position de l'ancien hub) ---
b5  = BioKan.Bacterium(5,  [ -1.0*s,    1.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa) # Pacemaker (dist 1s à B3 et B4)

# --- COUCHE 3 : Pré-sortie ---
b6  = BioKan.Bacterium(6,  [  0.0*s,    0.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa) # pre_out  (dist 1s à B3, B4, B7)

# --- COUCHE 4 : Sortie ---
b7  = BioKan.Bacterium(7,  [  0.0*s,   -1.0*s,  0.0], circuit, params_dict, u0_dict; mode=:ssa) # Output   (dist 1s à B6)

for b in [b1,b2,b3,b4,b5,b6,b7]
    BioKan.add_bacterium!(net, b)
end
BioKan.build_edges!(net)

n_bac = length(net.nodes)

BioKan.set_species!(net.nodes[1],  :S, 100.0)  # B1  Input_A
BioKan.set_species!(net.nodes[2],  :S, 100.0)  # B2  Input_B
BioKan.set_species!(net.nodes[3],  :S, 100.0)  # B3  relay_L
BioKan.set_species!(net.nodes[4],  :S, 100.0)  # B4  relay_R
BioKan.set_species!(net.nodes[5],  :S, 100.0)  # B5  Pacemaker
BioKan.set_species!(net.nodes[6],  :S, 100.0)  # B6  pre_out
BioKan.set_species!(net.nodes[7],  :S, 100.0)  # B7  Output
=#
@printf "Réseau : %d bactéries  |  spacing=%.2e m  |  d_max=%.2e m\n" n_bac spacing d_max

# Poids pour toutes les espèces
weights_all = BioKan.compute_static_coupling_physics(
    net.edges, D_dict, gamma_dict, species_names, R_cell, dt)

instant_set = (:O_diff, :E_ext_diff)

# Remapper les clés (id_s, id_t) → (bi_s, bi_t) car Diffusion.jl utilise les clés
# comme indices de tableau (1:n_bac). Nécessaire quand les IDs ne sont pas 1:n_bac.
let sorted_ids = sort(collect(keys(net.nodes)))
    global id_to_bi = Dict(id => bi for (bi, id) in enumerate(sorted_ids))
end
remap_w(w) = Dict((id_to_bi[k[1]], id_to_bi[k[2]]) => copy(v) for (k, v) in w)

weights_delayed = remap_w(weights_all)
for v in values(weights_delayed)
    for sp in instant_set
        v[findfirst(==(sp), species_names)] = 0.0
    end
end

weights_instant = remap_w(weights_all)
for v in values(weights_instant)
    for (idx, sp) in enumerate(species_names)
        sp ∉ instant_set && (v[idx] = 0.0)
    end
end

# Buffer circulaire pour le délai fixe (900 s avec dt=1)
delay_steps  = 90
delay_buffer = zeros(Float64, n_bac, n_species, delay_steps)
delay_ptr    = 1

# 4. PROTOCOLE — PORTE LOGIQUE
Random.seed!(43)
logical_gate      = length(ARGS) >= 1 ? Symbol(ARGS[1]) : :XOR
n_epochs          = 300
n_test_epochs     = 50
n_pause_epochs    = 0
max_density       = 120.0
stimulus_duration = 2000.0
interval          = 25000.0
delay_io          = 7000.0
force_duration    = 2000.0
eval_duration     = 20000.0

time_steps, Input_A, Input_B, Input_0, Output_forced, O_silent_A, O_silent_B, O_silent_output, mask, train_mask, freeze_intervals =
    BioKan.pattern_to_learn_xor(;
        logical_gate      = logical_gate,
        epochs            = n_epochs,
        test_epochs       = n_test_epochs,
        pause_epochs      = n_pause_epochs,
        dt                = dt,
        max_density       = max_density,
        stimulus_duration = stimulus_duration,
        interval          = interval,
        delay_io          = delay_io,
        force_duration    = force_duration,
        eval_duration     = eval_duration)
total_steps = length(time_steps)
time_vec    = collect(time_steps)

@printf "Protocole : %d trials  |  steps=%d  |  T_total=%.0fs\n" length(mask) total_steps time_steps[end]
@printf "Phases de gel de S : %d fenêtre(s)\n" length(freeze_intervals)
for (i, fi) in enumerate(freeze_intervals)
    @printf "  Gel %d : t=%.0f → %.0fs\n" i fi.t_start fi.t_end
end
flush(stdout)

# 5. PARAMÈTRES DU SPIKE THRESHOLD + PACEMAKER
M_thr_spike  = 97.0
M_thr_O      = 160.0
t_refrac      = 10000.0
t_spike_emit  = 2000.0
t_spike_emit_E = 200.0
steps_refrac  = round(Int, t_refrac       / dt)
steps_spike   = round(Int, t_spike_emit   / dt)
steps_spike_E = round(Int, t_spike_emit_E / dt)

spike_amp_sigma      = 0.0
spike_duration_sigma = 200.0

use_s_rescue   = false
freeze_relays  = false  # true → S de B1=relay_A et B2=relay_B figés à 150 comme le hub
s_rescue_val   = 100.0
steps_s_rescue = round(Int, 3e5 / dt)

freeze_s_mean  = true
n_s_mean       = 150

amp_D_diff     = round(Int, 3200  * dt)
amp_E_ext_diff = round(Int, 10000 * dt)
amp_C_diff     = round(Int, 3000  * dt)
amp_E_int      = round(Int, 3000  * dt)
amp_O          = round(Int, 3000  * dt)

t_o_inhib      = 10000.0 #5500
steps_o_inhib  = round(Int, t_o_inhib / dt)

output_silent_O  = true
input_silent_O   = true
input_silent_C   = false

# --- Pacemaker  ---
pacemaker_bi = 8   # bi=8 = B8=Pacemaker (IDs 1..10 consécutifs)

println(output_silent_O ? "Output silent O : activé\n" : "Output silent O : désactivé\n")
println(input_silent_O  ? "Input silent O : activé\n"  : "Input silent O : désactivé\n")
@printf "Pacemaker B8 (bi=%d) : spike synchronisé aux inputs (B1 ou B2) | B5=hub S figé à 150\n" pacemaker_bi
println("Spike : M>$(M_thr_spike) → $(t_spike_emit)s d'émission ($(amp_D_diff)/step) | M>$(M_thr_O) → O | réfractaire=$(t_refrac)s\n")

# 6. BOUCLE DE SIMULATION
history = zeros(Float16, total_steps, n_bac, n_species)

ref_b        = net.nodes[1]
sp_idx       = Int[get(ref_b.species_index, species_names[sp], 0) for sp in 1:n_species]
diffusible   = [D_dict[species_names[sp]] > 1e-40        for sp in 1:n_species]
decay_vec    = [exp(-gamma_dict[species_names[sp]] * dt) for sp in 1:n_species]
target_vec   = [diffusion_targets[species_names[sp]]     for sp in 1:n_species]

diff_list      = [sp for sp in 1:n_species if diffusible[sp]]
diff_src_idx   = Int[get(ref_b.species_index, species_names[sp], 0) for sp in diff_list]
diff_tgt_idx   = Int[get(ref_b.species_index, target_vec[sp], 0)    for sp in diff_list]
diff_decay     = [decay_vec[sp] for sp in diff_list]
o_diff_li      = findfirst(i -> species_names[diff_list[i]] == :O_diff, 1:length(diff_list))

d_inj        = sp_idx[findfirst(==(:D),         species_names)]
m_idx_sp     = sp_idx[findfirst(==(:M),         species_names)]
o_idx_sp     = sp_idx[findfirst(==(:O),         species_names)]
d_diff_idx   = sp_idx[findfirst(==(:D_diff),    species_names)]
e_diff_idx   = sp_idx[findfirst(==(:E_ext_diff),species_names)]
c_diff_idx   = sp_idx[findfirst(==(:C_diff),    species_names)]
o_diff_idx   = sp_idx[findfirst(==(:O_diff),    species_names)]
e_int_idx_sp = sp_idx[findfirst(==(:E_int),     species_names)]
s_idx_sp     = sp_idx[findfirst(==(:S),         species_names)]
s_hist_idx   = findfirst(==(:S), species_names)

flux_emissions = zeros(Float64, n_bac, n_species)
received       = zeros(Float64, n_bac, n_species)

refrac_counter    = zeros(Int, n_bac)
emit_counter      = zeros(Int, n_bac)
emit_E_counter    = zeros(Int, n_bac)
emit_Eint_counter = zeros(Int, n_bac)
emit_C_counter    = zeros(Int, n_bac)
emit_amp_D_vec    = fill(amp_D_diff, n_bac)
o_inhib_counter   = zeros(Int, n_bac)
s_zero_counter    = zeros(Int, n_bac)
spike_logs        = [Float32[] for _ in 1:n_bac]
cancelled_logs    = [Float32[] for _ in 1:n_bac]

# IDs 1..10 consécutifs → bi = ID
valid_nodes = [net.nodes[i] for i in 1:n_bac]

node_labels = ["B1=relay_A", "B2=relay_B", "B3=relay_L", "B4=relay_R",
               "B5=hub",     "B6=pre_out", "B7=Output",
               "B8=Pacemaker", "B9=Input_A", "B10=Input_B"]

s_frozen    = zeros(Float64, n_bac)
in_freeze   = false
freeze_idx  = 1

# bi=7=B7(Output) bi=8=B8(Pacemaker) bi=9=B9(Input_A) bi=10=B10(Input_B)
# B5=hub (bi=5) non dans forced_only : figé séparément en permanence
forced_only  = Set([7, 8, 9, 10])   # bloqués pendant entraînement
inputs_only  = Set([8, 9, 10])      # bloqués aussi en test

# Carte des voisins (bi = ID pour IDs consécutifs)
neighbors_of = [Int[] for _ in 1:n_bac]
for (ia, ic, _) in net.edges
    ia == ic && continue
    push!(neighbors_of[ia], ic)
end
abort_self_O  = zeros(Int, n_bac)   # émission avortée : O venait de soi (M>M_thr_O)
abort_neigh_O = zeros(Int, n_bac)   # ...               : O reçu d'un voisin
abort_latched = zeros(Int, n_bac)   # ...               : verrou résiduel (O déjà décru)

t_start = time()
for step in 1:total_steps

    # A. INJECTION  (Input_A=B9, Input_B=B10, Output=B7, Pacemaker=B8)
    if Input_A[step] > 0.0
        net.nodes[9].integrator.u[d_inj] = max(0, round(Int, Input_A[step] * dt + randn() * spike_amp_sigma))
        BioKan.notify_bacterium!(net.nodes[9])
    elseif input_silent_O && O_silent_A[step] > 0.0
        net.nodes[9].integrator.u[o_diff_idx] = amp_O
        BioKan.notify_bacterium!(net.nodes[9])
    end
    if Input_B[step] > 0.0
        net.nodes[10].integrator.u[d_inj] = max(0, round(Int, Input_B[step] * dt + randn() * spike_amp_sigma))
        BioKan.notify_bacterium!(net.nodes[10])
    elseif input_silent_O && O_silent_B[step] > 0.0
        net.nodes[10].integrator.u[o_diff_idx] = amp_O
        BioKan.notify_bacterium!(net.nodes[10])
    end
    if Output_forced[step] > 0.0
        net.nodes[7].integrator.u[d_inj] = max(0, round(Int, Output_forced[step] * dt + randn() * spike_amp_sigma))
        BioKan.notify_bacterium!(net.nodes[7])
    elseif output_silent_O && !in_freeze && O_silent_output[step] > 0.0
        net.nodes[7].integrator.u[o_diff_idx] = amp_O
        BioKan.notify_bacterium!(net.nodes[7])
    end
    # B8 = pacemaker : injection D au début de chaque trial (comme un input systématique)
    if Input_A[step] > 0.0 || Input_B[step] > 0.0 || Input_0[step] > 0.0
        net.nodes[8].integrator.u[d_inj] = round(Int, max_density * dt)  # B8=Pacemaker
        BioKan.notify_bacterium!(net.nodes[8])
    end

    # B. BIOLOGIE INTERNE
    for b in valid_nodes
        BioKan.step_bacterium!(b, dt)
    end

    # B5 = hub : S figé en permanence (entraînement ET test)
    net.nodes[5].integrator.u[s_idx_sp] = hub_s_frozen
    BioKan.notify_bacterium!(net.nodes[5])
    # B1=relay_A, B2=relay_B : S figé si freeze_relays=true
    if freeze_relays
        net.nodes[1].integrator.u[s_idx_sp] = relay_s_frozen
        BioKan.notify_bacterium!(net.nodes[1])
        net.nodes[2].integrator.u[s_idx_sp] = relay_s_frozen
        BioKan.notify_bacterium!(net.nodes[2])
    end

    # B'. GEL DE S pendant la phase de test
    t_now = time_steps[step]
    if freeze_idx <= length(freeze_intervals)
        fi = freeze_intervals[freeze_idx]
        if !in_freeze && t_now >= fi.t_start
            global in_freeze = true
            if freeze_s_mean && !isempty(train_mask)
                n_avg = min(n_s_mean * 4, length(train_mask))
                last_trials = train_mask[end-n_avg+1:end]
                for (bi, b) in enumerate(valid_nodes)
                    if bi in forced_only
                        s_frozen[bi] = 150.0
                    else
                        s_vals = Float64[]
                        for tr in last_trials
                            st = clamp(round(Int, tr.t_end / dt), 1, step - 1)
                            push!(s_vals, Float64(history[st, bi, s_hist_idx]))
                        end
                        s_frozen[bi] = mean(s_vals)
                    end
                end
            else
                for (bi, b) in enumerate(valid_nodes)
                    s_frozen[bi] = (bi in forced_only) ? 150.0 : Float64(b.integrator.u[s_idx_sp])
                end
            end
            s_last = [Float64(valid_nodes[bi].integrator.u[s_idx_sp]) for bi in 1:n_bac]
            @printf "Gel S #%d démarré à t=%.0fs (mode=%s, n=%d epochs)\n" freeze_idx t_now (freeze_s_mean ? "mean" : "last") (freeze_s_mean ? min(n_s_mean, div(length(train_mask), 4)) : 1)
            @printf "  %-20s  %8s  %8s  %8s\n" "Bactérie" "last" "frozen" "diff"
            for bi in 1:n_bac
                @printf "  %-20s  %8.1f  %8.1f  %+8.1f\n" node_labels[bi] s_last[bi] s_frozen[bi] (s_frozen[bi] - s_last[bi])
            end
            flush(stdout)
        end
        if in_freeze && t_now >= fi.t_end
            global in_freeze = false
            global freeze_idx += 1
            @printf "Gel S #%d terminé à t=%.0fs\n" (freeze_idx - 1) t_now
        end
    end
    if in_freeze
        for (bi, b) in enumerate(valid_nodes)
            b.integrator.u[s_idx_sp] = round(Int, clamp(s_frozen[bi], 0.0, 1e6))
        end
    end

    # C. LOGIQUE DE SPIKE
    for (bi, b) in enumerate(valid_nodes)
        M_val = b.integrator.u[m_idx_sp]
        O_val = b.integrator.u[o_idx_sp]

        if O_val >= 1.0
            o_inhib_counter[bi] = steps_o_inhib
        elseif o_inhib_counter[bi] > 0
            o_inhib_counter[bi] -= 1
        end
        is_o_inhibited = o_inhib_counter[bi] > 0

        # --- LOGIQUE NORMALE (tous les nœuds, pacemaker inclus) ---
        if M_val >= M_thr_spike && is_o_inhibited && refrac_counter[bi] == 0 && emit_counter[bi] == 0
            push!(cancelled_logs[bi], Float32(step * dt))
        end

        if M_val >= M_thr_spike && !is_o_inhibited && refrac_counter[bi] == 0 && emit_counter[bi] == 0
            any_trial = Input_A[step] > 0.0 || Input_B[step] > 0.0 || Input_0[step] > 0.0
            can_spike = (in_freeze && !(bi in inputs_only)) ||
                        !(bi in forced_only) ||
                        (bi == 8  && any_trial) ||              # bi=8=B8=Pacemaker
                        (bi == 9  && Input_A[step]  > 0.0) ||  # bi=9=B9=Input_A
                        (bi == 10 && Input_B[step]  > 0.0) ||  # bi=10=B10=Input_B
                        (bi == 7  && Output_forced[step] > 0.0) # bi=7=B7=Output
            if can_spike
                steps_spike_i        = max(1, round(Int, steps_spike + randn() * spike_duration_sigma / dt))
                emit_counter[bi]     = steps_spike_i
                emit_E_counter[bi]   = steps_spike_E
                emit_Eint_counter[bi]  = steps_spike
                emit_C_counter[bi]     = steps_spike
                refrac_counter[bi]    = steps_spike_i + steps_refrac
                push!(spike_logs[bi], Float32(step * dt))
                emit_amp_D_vec[bi] = max(0, round(Int, amp_D_diff + randn() * spike_amp_sigma))
            end
        end

        if emit_counter[bi] > 0
            if is_o_inhibited
                # O avorte l'émission (D/E_ext/C) mais ne coupe PLUS E_int
                if b.integrator.u[o_idx_sp] >= 1.0
                    abort_self_O[bi] += 1
                elseif any(nj -> valid_nodes[nj].integrator.u[o_idx_sp] >= 1.0, neighbors_of[bi])
                    abort_neigh_O[bi] += 1
                else
                    abort_latched[bi] += 1
                end
                refrac_counter[bi]    = max(refrac_counter[bi], steps_spike + steps_refrac)
                emit_counter[bi]      = 0
                emit_E_counter[bi]    = 0
                emit_Eint_counter[bi] = 0
                emit_C_counter[bi]    = 0
                b.integrator.u[d_diff_idx]   = 0
                b.integrator.u[e_diff_idx]   = 0
                b.integrator.u[c_diff_idx]   = 0
                b.integrator.u[e_int_idx_sp] = 0
            else
                b.integrator.u[d_diff_idx]   = (bi == 7) ? 0 : emit_amp_D_vec[bi]   # bi=7=B7=Output n'émet pas D_diff
                b.integrator.u[e_diff_idx]   = emit_E_counter[bi] > 0 ? amp_E_ext_diff : 0
                b.integrator.u[c_diff_idx]   = (bi == 7 || (input_silent_C && (bi == 9 || bi == 10))) ? 0 : (emit_C_counter[bi] > 0 ? amp_C_diff : 0)
                b.integrator.u[e_int_idx_sp] = emit_Eint_counter[bi] > 0 ? amp_E_int : 0
                if M_val >= M_thr_O && !(bi in forced_only)
                    b.integrator.u[o_diff_idx]   = emit_amp_D_vec[bi]
                end
                emit_counter[bi]      -= 1
                emit_E_counter[bi]     = max(0, emit_E_counter[bi] - 1)
                emit_Eint_counter[bi]  = max(0, emit_Eint_counter[bi] - 1)
                emit_C_counter[bi]     = max(0, emit_C_counter[bi] - 1)
            end
        end

        refrac_counter[bi] > 0 && (refrac_counter[bi] -= 1)

        if use_s_rescue && !in_freeze
            if b.integrator.u[s_idx_sp] == 0
                s_zero_counter[bi] += 1
                if s_zero_counter[bi] >= steps_s_rescue
                    b.integrator.u[s_idx_sp] = s_rescue_val
                    s_zero_counter[bi] = 0
                end
            else
                s_zero_counter[bi] = 0
            end
        end
    end

    # D. FLUX DIFFUSIBLES
    fill!(flux_emissions, 0.0)
    for (bi, b) in enumerate(valid_nodes)
        for (li, sp) in enumerate(diff_list)
            v = b.integrator.u[diff_src_idx[li]]
            v > 0.0 && (flux_emissions[bi, sp] = v)
        end
    end

    # E. TRANSPORT (D/E/C avec délai 900s, O instantané)
    global received, delay_ptr
    received, delay_ptr = BioKan.propagate_signals_delayed!(
        received, weights_delayed, weights_instant,
        flux_emissions, delay_buffer, delay_ptr, delay_steps, n_species)

    # F. UPDATE DIFFUSION
    for (bi, b) in enumerate(valid_nodes)
        for (li, sp) in enumerate(diff_list)
            amt = round(Int, max(0.0, received[bi, sp] * diff_decay[li]))
            amt > 0 && (b.integrator.u[diff_tgt_idx[li]] += amt)
            b.integrator.u[diff_src_idx[li]] = 0
        end
        BioKan.notify_bacterium!(b)
    end

    # G. ENREGISTREMENT
    for (bi, b) in enumerate(valid_nodes)
        for sp in 1:n_species
            history[step, bi, sp] = Float16(max(0.0, b.integrator.u[sp_idx[sp]]))
        end
    end
end
@printf "Simulation terminée en %.1f s\n" (time() - t_start)
for bi in 1:n_bac
    @printf "Spikes B%d : %d  |  annulés par O : %d\n" bi length(spike_logs[bi]) length(cancelled_logs[bi])
end

# Source du O qui AVORTE une émission en cours (coupe E_int prématurément)
println("\n--- Émissions avortées par O : d'où vient le O ? ---")
@printf "  %-20s  %8s  %8s  %8s  %8s\n" "Bactérie" "total" "soi" "voisin" "verrou"
for bi in 1:n_bac
    tot = abort_self_O[bi] + abort_neigh_O[bi] + abort_latched[bi]
    tot == 0 && continue
    @printf "  %-20s  %8d  %8d  %8d  %8d\n" node_labels[bi] tot abort_self_O[bi] abort_neigh_O[bi] abort_latched[bi]
end
println("  soi = M>M_thr_O ici | voisin = O reçu d'un voisin | verrou = O déjà décru, inhibition résiduelle (≤t_o_inhib)")
flush(stdout)

# 7. DIAGNOSTICS

idx_sp = Dict(name => findfirst(==(name), species_names) for name in species_names)

# --- S[B5=hub] par trial d'entraînement ---
println("\n--- S[B5=hub] fin de chaque trial d'entraînement ---")
@printf "  %-6s  %-8s  %-10s  %-8s\n" "trial" "combo" "t_end(s)" "S_B5"
junction_bi = 5
for (ti, tr) in enumerate(train_mask)
    step_end = clamp(round(Int, tr.t_end / dt), 1, total_steps)
    s_val    = history[step_end, junction_bi, idx_sp[:S]]
    @printf "  %-6d  %-8s  %-10.0f  %.1f\n" ti tr.combo tr.t_end s_val
end
println("\n  Résumé S[B5=hub] moyen par combo (entraînement) :")
for combo in [:A_only, :B_only, :AB, :none]
    vals = [Float64(history[clamp(round(Int, tr.t_end/dt),1,total_steps), junction_bi, idx_sp[:S]])
            for tr in train_mask if tr.combo == combo]
    isempty(vals) && continue
    @printf "    %-8s  n=%d  mean=%.1f  min=%.1f  max=%.1f\n" combo length(vals) mean(vals) minimum(vals) maximum(vals)
end

# --- Durée effective de E_int après chaque spike ---
println("\n--- Durée effective E_int après spike (seuil = amp_E_int/10) ---")
e_int_thr = amp_E_int / 10.0
for bi in 1:n_bac
    durations = Float64[]
    for ts in spike_logs[bi]
        step_start = clamp(round(Int, Float64(ts) / dt), 1, total_steps)
        step_end_search = min(step_start + round(Int, 20000.0 / dt), total_steps)
        dur = 0.0
        for st in step_start:step_end_search
            if history[st, bi, idx_sp[:E_int]] >= e_int_thr
                dur += dt
            else
                dur > 0.0 && break
            end
        end
        dur > 0.0 && push!(durations, dur)
    end
    isempty(durations) && continue
    @printf "  B%d %-14s  n=%d  mean=%.0fs  min=%.0fs  max=%.0fs  std=%.0fs\n" bi node_labels[bi] length(durations) mean(durations) minimum(durations) maximum(durations) std(durations)
end

# --- Durée de présence E_ext / E_int / C après chaque spike (par nœud) ---
println("\n--- Durée de présence E_ext / E_int / C après chaque spike (par nœud) ---")
@printf "  %-20s  %6s  %7s  %10s  %10s  %10s  %12s\n" "Bactérie" "n_spk" "%coinc" "E_ext(s)" "E_int(s)" "C(s)" "coinc_EE(s)"
search_window = round(Int, 25000.0 / dt)
thr_frac = 10.0
thr_Ee = 1.0
thr_Ei = amp_E_int / thr_frac
thr_C  = 1.0
for bi in 1:n_bac
    dur_Ee = Float64[]; dur_Ei = Float64[]; dur_C = Float64[]; dur_coinc = Float64[]
    for ts in spike_logs[bi]
        st0 = clamp(round(Int, Float64(ts) / dt), 1, total_steps)
        st1 = min(st0 + search_window, total_steps)
        for (dur_vec, sp_idx, thr) in (
                (dur_Ee, idx_sp[:E_ext], thr_Ee),
                (dur_Ei, idx_sp[:E_int], thr_Ei),
                (dur_C,  idx_sp[:C],     thr_C))
            d = 0.0
            for st in st0:st1
                if history[st, bi, sp_idx] >= thr
                    d += dt
                else
                    d > 0.0 && break
                end
            end
            d > 0.0 && push!(dur_vec, d)
        end
        # coïncidence : E_ext ET E_int présents simultanément
        d_coinc = 0.0
        for st in st0:st1
            if history[st, bi, idx_sp[:E_ext]] >= thr_Ee && history[st, bi, idx_sp[:E_int]] >= thr_Ei
                d_coinc += dt
            end
        end
        d_coinc > 0.0 && push!(dur_coinc, d_coinc)
    end
    n_spk = length(spike_logs[bi])
    n_spk == 0 && continue
    me = isempty(dur_Ee)    ? 0.0 : mean(dur_Ee)
    mi = isempty(dur_Ei)    ? 0.0 : mean(dur_Ei)
    mc = isempty(dur_C)     ? 0.0 : mean(dur_C)
    mco = isempty(dur_coinc) ? 0.0 : mean(dur_coinc)
    pco = 100.0 * length(dur_coinc) / n_spk
    @printf "  %-20s  %6d  %6.0f%%  %10.0f  %10.0f  %10.0f  %12.0f\n" node_labels[bi] n_spk pco me mi mc mco

    # Détail par spike
    @printf "    %-6s  %12s  %10s  %10s  %10s  %12s\n" "spike" "t(s)" "E_ext(s)" "E_int(s)" "C(s)" "coinc_EE(s)"
    for (si, ts) in enumerate(spike_logs[bi])
        st0 = clamp(round(Int, Float64(ts) / dt), 1, total_steps)
        st1 = min(st0 + search_window, total_steps)
        local_dur = Float64[0.0, 0.0, 0.0, 0.0]
        for (ki, (sp_idx, thr)) in enumerate((
                (idx_sp[:E_ext], thr_Ee),
                (idx_sp[:E_int], thr_Ei),
                (idx_sp[:C],     thr_C)))
            d = 0.0
            for st in st0:st1
                if history[st, bi, sp_idx] >= thr
                    d += dt
                else
                    d > 0.0 && break
                end
            end
            local_dur[ki] = d
        end
        d_coinc = 0.0
        for st in st0:st1
            if history[st, bi, idx_sp[:E_ext]] >= thr_Ee && history[st, bi, idx_sp[:E_int]] >= thr_Ei
                d_coinc += dt
            end
        end
        local_dur[4] = d_coinc
        @printf "    %-6d  %12.0f  %10.0f  %10.0f  %10.0f  %12.0f\n" si Float64(ts) local_dur[1] local_dur[2] local_dur[3] local_dur[4]
    end
end

# --- Sélectivité aux entrées : un spike suit-il une entrée, ou le pacemaker seul ? ---
# Pour chaque spike, on regarde si Input_A / Input_B étaient actifs dans les `lookback` s
# précédant le spike. "AUCUNE_entrée" = spike sans aucune entrée → tiré par le pacemaker.
println("\n--- Sélectivité aux entrées (look-back = 5000s avant chaque spike) ---")
@printf "  %-20s  %6s  %8s  %8s  %8s  %14s\n" "Bactérie" "n_spk" "A_actif" "B_actif" "A+B" "AUCUNE_entrée"
lookback = round(Int, 5000.0 / dt)
for bi in 1:n_bac
    n_spk = length(spike_logs[bi])
    n_spk == 0 && continue
    cA = 0; cB = 0; cAB = 0; cNone = 0
    for ts in spike_logs[bi]
        st = clamp(round(Int, Float64(ts) / dt), 1, total_steps)
        w0 = max(1, st - lookback)
        a = any(x -> x > 0.0, view(Input_A, w0:st))
        b = any(x -> x > 0.0, view(Input_B, w0:st))
        if a && b
            cAB += 1
        elseif a
            cA += 1
        elseif b
            cB += 1
        else
            cNone += 1
        end
    end
    @printf "  %-20s  %6d  %8d  %8d  %8d  %14d\n" node_labels[bi] n_spk cA cB cAB cNone
end
println("  (AUCUNE_entrée élevé sur un relais = il décharge sur le pacemaker, pas sur son entrée → entrée ignorée)")

# --- Rapport XOR ---
println("\n--- Rapport $(logical_gate) (B7=Output) ---")
combo_labels = Dict(:A_only => "A seul", :B_only => "B seul", :AB => "A+B  ", :none => "aucun")
O_veto = 3.0
t_veto = 3000.0

trial_results = NamedTuple{(:trial, :combo, :expected, :spiked, :correct, :raw, :val, :t_start, :t_end)}[]
n_correct_global = 0
n_total_global   = 0

output_bi = 7   # bi=7=B7=Output
begin
    for (ti, trial) in enumerate(mask)
        spikes_raw = filter(t -> trial.t_start <= t <= trial.t_end, spike_logs[output_bi])
        spikes_validated = count(spikes_raw) do ts
            step_spike = clamp(round(Int, Float64(ts) / dt), 1, total_steps)
            step_end   = min(step_spike + round(Int, t_veto / dt), total_steps)
            maximum(history[step_spike:step_end, output_bi, idx_sp[:O]]) < O_veto
        end
        spiked   = spikes_validated > 0
        expected = trial.expected == 1
        correct  = spiked == expected
        global n_correct_global += correct
        global n_total_global   += 1
        label = correct ? "✓" : "✗"
        combo = get(combo_labels, trial.combo, string(trial.combo))
        @printf "  %s [trial %2d | %s] exp=%d  raw=%d  val=%d  t=[%.0f,%.0f]s\n" label ti combo trial.expected length(spikes_raw) spikes_validated trial.t_start trial.t_end
        push!(trial_results, (trial=ti, combo=trial.combo, expected=Int(trial.expected),
                              spiked=Int(spiked), correct=Int(correct),
                              raw=length(spikes_raw), val=spikes_validated,
                              t_start=trial.t_start, t_end=trial.t_end))
    end
    @printf "\n  Score : %d / %d (%.0f%%)\n" n_correct_global n_total_global 100*n_correct_global/max(1,n_total_global)
end

println("\n--- Max par espèce ---")
for (sp, name) in enumerate(species_names)
    vals = [maximum(history[:, bi, sp]) for bi in 1:n_bac]
    any(v -> v > 0.01, vals) && @printf "  %-14s %s\n" name join([@sprintf("%s=%.2f", node_labels[bi], vals[bi]) for bi in 1:n_bac], "   ")
end

# 8. TRACÉ
t_vec      = collect(time_vec)
K_MD_plot  = 250.0
plot_stride = max(1, div(length(t_vec), 5000))
tp = t_vec[1:plot_stride:end]

function downsample_max(trace, stride)
    stride == 1 && return trace
    return [maximum(view(trace, i:min(i+stride-1, length(trace)))) for i in 1:stride:length(trace)]
end

function add_eval_windows!(p)
    for trial in mask
        col = trial.expected == 1 ? RGBA(0.0,0.6,0.0,0.15) : RGBA(1.0,0.5,0.0,0.15)
        vspan!(p, [trial.t_start, trial.t_end], color=col, label="")
    end
    vline!(p, [-1.0], color=RGBA(0.0,0.6,0.0,0.6), lw=2, label="eval exp=1")
    vline!(p, [-1.0], color=RGBA(1.0,0.5,0.0,0.6), lw=2, label="eval exp=0")
end

function add_spikes!(p, spk_log, col)
    for ts in spk_log
        ts <= maximum(tp) && vline!(p, [Float64(ts)], color=col, lw=0.5, alpha=0.5, label="")
    end
end

function add_cancellations!(p, can_log, y_marker; bin_size=2*t_refrac)
    isempty(can_log) && return
    t_max = maximum(tp)
    last_t = -Inf
    first_added = false
    for ts in sort(can_log)
        Float64(ts) > t_max && continue
        if Float64(ts) - last_t >= bin_size
            scatter!(p, [Float64(ts)], [y_marker],
                     markershape=:dtriangle, markersize=6,
                     markercolor=:red, markerstrokewidth=0,
                     label=(first_added ? "" : "annulé par O"), alpha=0.8)
            last_t = Float64(ts)
            first_added = true
        end
    end
end

inp_stride = plot_stride
tp_full    = time_vec[1:inp_stride:end]
inA_ds     = Input_A[1:inp_stride:length(tp_full)*inp_stride]
inB_ds     = Input_B[1:inp_stride:length(tp_full)*inp_stride]
in0_ds     = Input_0[1:inp_stride:length(tp_full)*inp_stride]
out_ds     = Output_forced[1:inp_stride:length(tp_full)*inp_stride]
n_tp       = min(length(tp_full), length(inA_ds), length(inB_ds))

p_input = plot(tp_full[1:n_tp], inA_ds[1:n_tp], label="Input_A (B1)", color=:blue, lw=1.5,
               title="Stimuli injectés", ylabel="mol", xlabel="Temps (s)",
               legend=:topleft, tickfontsize=10, guidefontsize=10)
plot!(p_input, tp_full[1:n_tp], inB_ds[1:n_tp], label="Input_B (B3)",       color=:orange, lw=1.5)
plot!(p_input, tp_full[1:n_tp], in0_ds[1:n_tp], label="Protocole Input_0 (ignoré B8 pace)", color=:gray, lw=1.0, ls=:dash)
plot!(p_input, tp_full[1:n_tp], out_ds[1:n_tp], label="Output forcé (B6)",  color=:teal,   lw=1.5)
add_eval_windows!(p_input)

panels_dmi  = [p_input]
panels_zoom = [p_input]

bac_colors = [:dodgerblue, :darkorange, :steelblue, :peru, :green, :mediumpurple, :crimson, :black, :blue, :orange]

for bi in 1:n_bac
    lab     = node_labels[bi]
    spk_col = bac_colors[bi]
    spk_log = spike_logs[bi]

    d_trace  = downsample_max(Float64.(history[:, bi, idx_sp[:D]]),     plot_stride)
    m_trace  = downsample_max(Float64.(history[:, bi, idx_sp[:M]]),     plot_stride)
    ei_trace = downsample_max(Float64.(history[:, bi, idx_sp[:E_int]]), plot_stride)
    ee_trace = downsample_max(Float64.(history[:, bi, idx_sp[:E_ext]]), plot_stride)
    c_trace  = downsample_max(Float64.(history[:, bi, idx_sp[:C]]),     plot_stride)
    s_trace  = downsample_max(Float64.(history[:, bi, idx_sp[:S]]),     plot_stride)
    o_trace  = downsample_max(Float64.(history[:, bi, idx_sp[:O]]),     plot_stride)

    @printf "\n[%s]\n" lab
    for (nm, tr) in [("D",d_trace),("M",m_trace),("E_int",ei_trace),("E_ext",ee_trace),
                     ("C",c_trace),("S",s_trace),("O",o_trace)]
        @printf "  %-8s max=%8.2f  mean=%6.3f\n" nm maximum(tr) mean(tr)
    end

    p1 = plot(tp, d_trace, label="D", color=:steelblue, lw=1,
              title=lab, ylabel="mol", xlabel="Temps (s)",
              legend=:topleft, tickfontsize=10, guidefontsize=10)
    plot!(p1, tp, m_trace, label="M", color=:crimson, lw=1)
    hline!(p1, [M_thr_spike], color=:red,    lw=1, ls=:dash, label="M_thr=$(M_thr_spike)")
    hline!(p1, [M_thr_O],     color=:orange, lw=1, ls=:dash, label="M_thr_O=$(M_thr_O)")
    add_eval_windows!(p1)
    add_spikes!(p1, spk_log, spk_col)
    add_cancellations!(p1, cancelled_logs[bi], M_thr_O * 0.9)
    push!(panels_dmi, p1)

    s_max = max(1.0, maximum(s_trace))
    p2 = plot(tp, s_trace, label="S", color=:black, lw=1.5,
              title="$(lab) — S", ylabel="mol", xlabel="Temps (s)",
              ylims=(0.0, s_max * 1.05), legend=false, tickfontsize=10, guidefontsize=10)
    add_eval_windows!(p2)
    add_spikes!(p2, spk_log, spk_col)
    push!(panels_zoom, p2)
end

ts_str  = Dates.format(now(), "yyyymmdd_HHMMSS")
n_dmi   = length(panels_dmi)
n_zoom  = length(panels_zoom)

fig_dmi  = plot(panels_dmi...,  layout=(n_dmi,  1), size=(1200, 400 * n_dmi))
fig_zoom = plot(panels_zoom..., layout=(n_zoom, 1), size=(1200, 400 * n_zoom))

panels_O = []
for bi in 1:n_bac
    o_trace = downsample_max(Float64.(history[:, bi, idx_sp[:O]]), plot_stride)
    d_trace_O = downsample_max(Float64.(history[:, bi, idx_sp[:D]]), plot_stride)
    p_O = plot(tp, o_trace, label="O", color=:crimson, lw=1.2,
               title=node_labels[bi], ylabel="mol", xlabel="Temps (s)",
               legend=:topleft, tickfontsize=10, guidefontsize=10)
    plot!(p_O, tp, d_trace_O, label="D", color=:steelblue, lw=1.0)
    hline!(p_O, [1.0], color=:black, lw=1, ls=:dot, label="seuil O=1")
    add_eval_windows!(p_O)
    add_spikes!(p_O, spike_logs[bi], bac_colors[bi])
    push!(panels_O, p_O)
end
fig_O = plot(panels_O..., layout=(n_bac, 1), size=(1200, 300 * n_bac))

panels_E = []
for bi in 1:n_bac
    ei_trace = downsample_max(Float64.(history[:, bi, idx_sp[:E_int]]), plot_stride)
    ee_trace = downsample_max(Float64.(history[:, bi, idx_sp[:E_ext]]), plot_stride)
    ei_cl = min.(ei_trace, 100.0)
    ee_cl = min.(ee_trace, 100.0)
    p_E = plot(tp, ei_cl, label="E_int", color=:brown, lw=1.2,
               title=node_labels[bi], ylabel="mol (max 100)", xlabel="Temps (s)",
               ylims=(0.0, 110.0), legend=:topleft, tickfontsize=10, guidefontsize=10)
    plot!(p_E, tp, ee_cl, label="E_ext", color=:pink, lw=1.2)
    add_eval_windows!(p_E)
    add_spikes!(p_E, spike_logs[bi], bac_colors[bi])
    push!(panels_E, p_E)
end
fig_E = plot(panels_E..., layout=(n_bac, 1), size=(1200, 300 * n_bac))

gate_str  = string(logical_gate)
batch_id  = length(ARGS) >= 2 ? ARGS[2] : ts_str
out_dir   = joinpath(PROJECT_ROOT, "outputs", "xorpace_$(batch_id)", gate_str)
mkpath(out_dir)
path_dmi  = joinpath(out_dir, "test_xorpace_$(ts_str)_DMI.png")
path_zoom = joinpath(out_dir, "test_xorpace_$(ts_str)_zoom.png")
path_O    = joinpath(out_dir, "test_xorpace_$(ts_str)_O.png")
path_E    = joinpath(out_dir, "test_xorpace_$(ts_str)_E.png")
savefig(fig_dmi,  path_dmi)
savefig(fig_zoom, path_zoom)
savefig(fig_O,    path_O)
savefig(fig_E,    path_E)
println("\nPlots sauvegardés :\n  $path_dmi\n  $path_zoom\n  $path_O\n  $path_E")

# 9. ZOOM PHASE DE TEST + TOPOLOGIE DU RÉSEAU

sorted_ids_net = sort(collect(keys(net.nodes)))
pos_x_net = [net.nodes[i].pos[1] for i in sorted_ids_net]
pos_y_net = [net.nodes[i].pos[2] for i in sorted_ids_net]
pos_z_net = [length(net.nodes[i].pos) >= 3 ? net.nodes[i].pos[3] : 0.0 for i in sorted_ids_net]
xy_pad = 1.5 * spacing
xz_pad = 1.5 * spacing

function make_topo_panel(pos1, pos2, title_str, xlabel_str, ylabel_str)
    pad = 1.5 * spacing
    p = plot(title = title_str, xlabel = xlabel_str, ylabel = ylabel_str,
             legend = false, tickfontsize = 8, guidefontsize = 9,
             aspect_ratio = :equal,
             xlims = (minimum(pos1) - pad, maximum(pos1) + pad),
             ylims = (minimum(pos2) - pad, maximum(pos2) + pad))
    for (id1, id2, _) in net.edges
        id1 == id2 && continue
        bi1, bi2 = id_to_bi[id1], id_to_bi[id2]
        plot!(p, [pos1[bi1], pos1[bi2]], [pos2[bi1], pos2[bi2]],
              color = :gray60, lw = 2.5, label = "")
    end
    scatter!(p, pos1, pos2,
             markersize = 22, markercolor = bac_colors,
             markerstrokecolor = :black, markerstrokewidth = 1.5, label = "")
    for (bi, id) in enumerate(sorted_ids_net)
        annotate!(p, pos1[bi], pos2[bi], text("B$id", :center, 7, :white))
    end
    for (i, lab) in enumerate(node_labels)
        scatter!(p, [NaN], [NaN],
                 markercolor = bac_colors[mod1(i, length(bac_colors))],
                 markerstrokecolor = :black, markersize = 10, label = lab)
    end
    return p
end

p_net_xy = make_topo_panel(pos_x_net, pos_y_net, "Topologie — vue XY", "x (m)", "y (m)")
p_net_xz = make_topo_panel(pos_x_net, pos_z_net, "Topologie — vue XZ", "x (m)", "z (m)")
p_net = plot(p_net_xy, p_net_xz, layout=(1,2), size=(1200, 500))

t_train_end    = isempty(freeze_intervals) ? time_steps[end] : freeze_intervals[1].t_start
step_train_end = clamp(round(Int, t_train_end / dt), 1, total_steps)
s_lines = ["S entraînement (t=0 → $(round(Int, t_train_end))s)", "",
           @sprintf("%-20s  %8s  %8s", "Bactérie", "mean(S)", "final(S)")]
for bi in 1:n_bac
    s_train = Float64.(history[1:step_train_end, bi, idx_sp[:S]])
    push!(s_lines, @sprintf("%-20s  %8.1f  %8.1f", node_labels[bi], mean(s_train), s_train[end]))
end
p_s_info = plot(framestyle=:none, xlims=(0,1), ylims=(0,1), legend=false,
                margin=5Plots.mm)
for (li, ln) in enumerate(reverse(s_lines))
    annotate!(p_s_info, 0.02, (li - 1) / length(s_lines), text(ln, :left, 8, :black, "Courier"))
end

panels_test = Any[p_net, p_s_info]
for (fi_i, fi) in enumerate(freeze_intervals)
    t_lo, t_hi = fi.t_start, fi.t_end
    idx_test   = findall(t -> t_lo <= t <= t_hi, tp)
    isempty(idx_test) && continue
    tp_test = tp[idx_test]

    p_t = plot(title  = "Phase test — M par bactérie  (t=$(round(Int,t_lo))–$(round(Int,t_hi)) s)",
               xlabel = "Temps (s)", ylabel = "M (mol)",
               legend = :topleft, tickfontsize = 8, guidefontsize = 9)
    for trial in mask
        trial.t_start >= t_lo && trial.t_start <= t_hi || continue
        col_w = trial.expected == 1 ? RGBA(0.0,0.6,0.0,0.2) : RGBA(1.0,0.5,0.0,0.2)
        vspan!(p_t, [trial.t_start, min(trial.t_end, t_hi)], color = col_w, label = "")
    end
    vline!(p_t, [-1.0], color = RGBA(0.0,0.6,0.0,0.6), lw = 2, label = "eval exp=1")
    vline!(p_t, [-1.0], color = RGBA(1.0,0.5,0.0,0.6), lw = 2, label = "eval exp=0")
    hline!(p_t, [M_thr_spike], color = :red, lw = 1, ls = :dash, label = "M_thr=$(M_thr_spike)")
    for bi in 1:n_bac
        col_bi = bac_colors[mod1(bi, length(bac_colors))]
        lab_bi = node_labels[bi]
        m_full = downsample_max(Float64.(history[:, bi, idx_sp[:M]]), plot_stride)
        n_safe = min(length(tp), length(m_full))
        m_test = m_full[1:n_safe][idx_test[idx_test .<= n_safe]]
        tp_safe = tp[idx_test[idx_test .<= n_safe]]
        plot!(p_t, tp_safe, m_test, label = lab_bi, color = col_bi, lw = 1.5)
        for ts in spike_logs[bi]
            (t_lo <= Float64(ts) <= t_hi) &&
                vline!(p_t, [Float64(ts)], color = col_bi, lw = 1.2, alpha = 0.7, label = "")
        end
    end
    push!(panels_test, p_t)
end

n_tp2 = length(panels_test)
fig_test = plot(panels_test..., layout = (n_tp2, 1), size = (1400, 420 * n_tp2))
path_test = joinpath(out_dir, "test_xorpace_$(ts_str)_test.png")
savefig(fig_test, path_test)
println("  $path_test")

# --- Fichier résultats texte ---
path_results = joinpath(out_dir, "test_xorpace_$(ts_str)_results.txt")
open(path_results, "w") do io
    println(io, "=== Résultats PACE $(logical_gate) — $(ts_str) ===\n")
    println(io, "Protocole : $(n_epochs) epochs entraînement | $(n_test_epochs) epochs test")
    @printf io "Score global : %d / %d (%.1f%%)\n\n" n_correct_global n_total_global 100*n_correct_global/max(1,n_total_global)

    println(io, "--- Stats par type de stimulus ---")
    @printf io "  %-10s  %6s  %6s  %8s\n" "combo" "n" "correct" "score(%)"
    for combo in [:A_only, :B_only, :AB, :none]
        rows = filter(r -> r.combo == combo, trial_results)
        isempty(rows) && continue
        n_c = length(rows)
        nc  = sum(r.correct for r in rows)
        @printf io "  %-10s  %6d  %6d  %8.1f\n" combo n_c nc 100.0*nc/n_c
    end

    println(io, "\n--- Détail par trial ---")
    @printf io "  %-6s  %-10s  %6s  %6s  %6s\n" "trial" "combo" "exp" "raw" "val"
    for r in trial_results
        mark = r.correct == 1 ? "✓" : "✗"
        @printf io "  %s %-5d  %-10s  %6d  %6d  %6d\n" mark r.trial r.combo r.expected r.raw r.val
    end

    println(io, "\n--- S entraînement (mean / final) ---")
    @printf io "  %-20s  %8s  %8s\n" "Bactérie" "mean(S)" "final(S)"
    t_train_end_r = isempty(freeze_intervals) ? time_steps[end] : freeze_intervals[1].t_start
    step_te = clamp(round(Int, t_train_end_r / dt), 1, total_steps)
    for bi in 1:n_bac
        s_tr = Float64.(history[1:step_te, bi, idx_sp[:S]])
        @printf io "  %-20s  %8.1f  %8.1f\n" node_labels[bi] mean(s_tr) s_tr[end]
    end

    println(io, "\n--- Paramètres de simulation ---")
    @printf io "  r = %.4g\n" r
    @printf io "  pacemaker_bi=%d  (synchronisé aux trials)\n" pacemaker_bi
    param_names = ["n","m","l","v_max_M","v_max_learn","v_ground","v_max_CS","v_max_ES",
                   "K_DM","K_SM","K_CS","K_EiS","K_EeS","K_act_S","K_inh_S","K_IS","K_DS",
                   "K_spike","K_spike_O","v_max_spike_D","v_max_spike_E","v_max_spike_C",
                   "v_max_spike_T","v_max_spike_O","k_transl_M","k_transl_S",
                   "k_deg_mRNA_M","k_deg_mRNA_S","k_deg_M","k_deg_E_int","k_deg_E_ext",
                   "k_deg_C","k_deg_T","k_deg_S","k_deg_D","k_deg_O","k_echo_int",
                   "v_max_diff_D","v_max_diff_E","v_max_diff_C","v_max_diff_O","K_D_diff","K_OD"]
    for (i, (name, val)) in enumerate(zip(param_names, θ₀))
        @printf io "  θ₀[%2d] %-16s = %.6g\n" i name val
    end
    println(io, "\n--- Paramètres spike / diffusion ---")
    @printf io "  n_epochs=%d  n_test_epochs=%d  interval=%.0fs  stimulus_duration=%.0fs\n" n_epochs n_test_epochs interval stimulus_duration
    @printf io "  M_thr_spike=%.1f  M_thr_O=%.1f  t_refrac=%.0fs  t_spike_emit=%.0fs  t_spike_emit_E=%.0fs\n" M_thr_spike M_thr_O t_refrac t_spike_emit t_spike_emit_E
    @printf io "  amp_D_diff=%d  amp_E_ext_diff=%d  amp_C_diff=%d  amp_E_int=%d  amp_O=%d\n" amp_D_diff amp_E_ext_diff amp_C_diff amp_E_int amp_O
    @printf io "  lambda=%.4g  lambda_C=%.4g  spacing=%.4g  d_max=%.4g\n" lambda lambda_C spacing d_max
    @printf io "  output_silent_O=%s  freeze_s_mean=%s\n" output_silent_O freeze_s_mean
end
println("  $path_results")

# --- Fichier summary global ---
summary_dir  = joinpath(PROJECT_ROOT, "outputs", "xorpace_$(batch_id)")
path_summary = joinpath(summary_dir, "summary.csv")
path_lock    = joinpath(summary_dir, ".summary.lock")

t_lock = time()
while isfile(path_lock) && (time() - t_lock) < 60
    sleep(0.5)
end
touch(path_lock)
try
    write_header = !isfile(path_summary)
    open(path_summary, "a") do csv
        if write_header
            println(csv, "gate,score_pct,n_correct,n_total,A_only_pct,B_only_pct,AB_pct,none_pct")
        end
        combo_scores = Dict{Symbol, String}()
        for combo in [:A_only, :B_only, :AB, :none]
            rows = filter(r -> r.combo == combo, trial_results)
            if isempty(rows)
                combo_scores[combo] = "NA"
            else
                nc = sum(r.correct for r in rows)
                combo_scores[combo] = @sprintf("%.1f", 100.0 * nc / length(rows))
            end
        end
        @printf csv "%s,%.1f,%d,%d,%s,%s,%s,%s\n" gate_str 100.0*n_correct_global/max(1,n_total_global) n_correct_global n_total_global combo_scores[:A_only] combo_scores[:B_only] combo_scores[:AB] combo_scores[:none]
    end
    println("  $path_summary (mis à jour)")
finally
    rm(path_lock, force=true)
end
