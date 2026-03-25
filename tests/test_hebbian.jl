# Import Packages
include("../src/BioKan.jl")
using .BioKan
using Catalyst, JumpProcesses, Distributions, Plots

function generate_burst!(b::Bacterium, species::Symbol, val::Float64, prob::Float64)
    # Tirage aléatoire (Bernoulli)
    if rand() < prob
        add_input!(b, species, val)
        return true
    end
    return false
end

# ==============================================================================
# 1. Paramètres généraux
# ==============================================================================

n_bacteries    = 3
distance_comm  = 0.1
dt             = 1/10
total_steps    = Int(5000.0 / dt)
R_cell         = 0.5e-6

# Nœuds d'entrée :
#   B1 reçoit des bursts sur D  (signal "pré-synaptique")
#   B3 reçoit des bursts sur E_ext (signal "post-synaptique")
#   → coïncidence sur B2 devrait potentialiser M (règle STDP)
input_node_D     = [1]
input_node_E_ext = []

# ==============================================================================
# 2. Construction du réseau
# ==============================================================================

net = BioNetwork(distance_comm, n_bacteries)

# 16 espèces du modèle Hebbian
species_names = [
    :D, :M, :I, :C, :E_int, :T, :E_ext,
    :D_diff, :C_diff, :E_ext_diff,
    :mRNA_D, :mRNA_M, :mRNA_I, :mRNA_C, :mRNA_T, :mRNA_E_ext
]
species_index = Dict(sym => s for (s, sym) in enumerate(species_names))
n_species = length(species_names)

# Espèces diffusibles → espèce cible dans la cellule réceptrice
diffusion_targets = Dict(
    :D         => :D,
    :M         => :M,
    :I         => :I,
    :C         => :C,
    :E_int     => :E_int,
    :T         => :T,
    :E_ext     => :E_ext,
    :D_diff    => :D,       # D_diff reçu → ajouté à D
    :C_diff    => :C,       # C_diff reçu → ajouté à C
    :E_ext_diff => :E_ext,  # E_ext_diff reçu → ajouté à E_ext
    :mRNA_D    => :mRNA_D,
    :mRNA_M    => :mRNA_M,
    :mRNA_I    => :mRNA_I,
    :mRNA_C    => :mRNA_C,
    :mRNA_T    => :mRNA_T,
    :mRNA_E_ext => :mRNA_E_ext
)

# Coefficients de diffusion : seuls les _diff diffusent
D_dict = Dict(
    :D          => 0.0,
    :M          => 0.0,
    :I          => 0.0,
    :C          => 0.0,
    :E_int      => 0.0,
    :T          => 0.0,
    :E_ext      => 0.0,
    :D_diff     => 1.0e-9,   # comme X_diff dans burst
    :C_diff     => 5.0e-10,  # plus lent (signal de consolidation)
    :E_ext_diff => 1.0e-9,
    :mRNA_D     => 0.0,
    :mRNA_M     => 0.0,
    :mRNA_I     => 0.0,
    :mRNA_C     => 0.0,
    :mRNA_T     => 0.0,
    :mRNA_E_ext => 0.0
)

# Dégradation extracellulaire
gamma_dict = Dict(
    :D          => 0.0,
    :M          => 0.0,
    :I          => 0.0,
    :C          => 0.0,
    :E_int      => 0.0,
    :T          => 0.0,
    :E_ext      => 0.0,
    :D_diff     => 2.9e-2,
    :C_diff     => 5.0e-2,
    :E_ext_diff => 2.9e-2,
    :mRNA_D     => 0.0,
    :mRNA_M     => 0.0,
    :mRNA_I     => 0.0,
    :mRNA_C     => 0.0,
    :mRNA_T     => 0.0,
    :mRNA_E_ext => 0.0
)

# ==============================================================================
# 3. Circuit et bactéries
# ==============================================================================

circuit, p_defaults_vec = create_hebbian_model(:node_generic)
params_dict = Dict(parameters(circuit) .=> p_defaults_vec)

u0_dict_raw = Dict(
    :D => 0.0, :M => 0.0, :I => 0.0, :C => 0.0,
    :E_int => 0.0, :T => 0.0, :E_ext => 0.0,
    :D_diff => 0.0, :C_diff => 0.0, :E_ext_diff => 0.0,
    :mRNA_D => 0.0, :mRNA_M => 0.0, :mRNA_I => 0.0,
    :mRNA_C => 0.0, :mRNA_T => 0.0, :mRNA_E_ext => 0.0
)
u0_dict = map_symbols_to_species(circuit, u0_dict_raw)

b1 = Bacterium(1, [0.003100, 0.005], circuit, params_dict, u0_dict; mode=:ssa)
b2 = Bacterium(2, [0.003102, 0.005], circuit, params_dict, u0_dict; mode=:ssa)
b3 = Bacterium(3, [0.003104, 0.005], circuit, params_dict, u0_dict; mode=:ssa)

add_bacterium!(net, b1)
add_bacterium!(net, b2)
add_bacterium!(net, b3)

build_edges!(net)
println("Réseau : $(length(net.nodes)) bactéries.")
println("Distance B1->B2 : $(net.edges[2])")

# ==============================================================================
# 4. Simulation
# ==============================================================================

weights_matrix = BioKan.compute_static_coupling_physics(net.edges, D_dict, gamma_dict, species_names, R_cell, dt)

# Paramètres des bursts
burst_prob_D     = 0.001   # probabilité par dt (1 burst ≈ toutes les 1000 min)
burst_prob_E_ext = 0.001
burst_intensity_D     = 60.0
burst_intensity_E_ext = 30.0

# Buffers
flux_emissions = zeros(Float64, n_bacteries, n_species)
retained_stock = zeros(Float64, n_bacteries, n_species)

# Historique [temps, bactérie, espèce]
history = zeros(total_steps, n_bacteries, n_species)

println("🚀 Démarrage de la simulation ($total_steps pas)...")

for step in 1:total_steps
    t_sim = step * dt

    # === A. INPUTS (BURSTS) ===
    for id in input_node_D
        if haskey(net.nodes, id)
            triggered = generate_burst!(net.nodes[id], :D, burst_intensity_D, burst_prob_D)
            if triggered
                println("⚡ Burst D  au temps $t_sim (nœud $id)")
            end
        end
    end
    for id in input_node_E_ext
        if haskey(net.nodes, id)
            triggered = generate_burst!(net.nodes[id], :E_ext, burst_intensity_E_ext, burst_prob_E_ext)
            if triggered
                println("⚡ Burst E_ext au temps $t_sim (nœud $id)")
            end
        end
    end

    # === B. BIOLOGIE INTERNE ===
    for (id, b) in net.nodes
        step_bacterium!(b, dt)
    end

    # === C. GESTION DES FLUX (Diffusion instantanée) ===
    fill!(flux_emissions, 0.0)
    fill!(retained_stock, 0.0)

    for i in 1:n_bacteries
        if haskey(net.nodes, i)
            b = net.nodes[i]
            for s in 1:n_species
                sym = species_names[s]
                qty   = max(0.0, BioKan.get_species(b, sym))
                D_val = get(D_dict, sym, 0.0)

                if D_val <= 1e-40
                    retained_stock[i, s] = qty
                else
                    # Tout sort (mode instantané)
                    flux_emissions[i, s] = qty
                    retained_stock[i, s] = 0.0
                end
            end
        end
    end

    # === D. TRANSPORT ===
    received_signals = BioKan.propagate_signals_instantaneous!(weights_matrix, flux_emissions, n_bacteries, n_species)

    # === E. UPDATE & DÉGRADATION EXTERNE ===
    for i in 1:n_bacteries
        if haskey(net.nodes, i)
            b = net.nodes[i]
            for s in 1:n_species
                sym   = species_names[s]
                gamma = get(gamma_dict, sym, 0.0)
                D_val = get(D_dict, sym, 0.0)

                if D_val <= 1e-40
                    history[step, i, s] = max(0.0, BioKan.get_species(b, sym))
                else
                    # Signal diffusible reçu → injecté dans l'espèce cible
                    target_sym  = diffusion_targets[sym]
                    amount_received = max(0.0, received_signals[i, s] * exp(-gamma * dt))

                    current_val = BioKan.get_species(b, target_sym)
                    BioKan.set_species!(b, target_sym, current_val + amount_received)
                    BioKan.set_species!(b, sym, 0.0)

                    history[step, i, s] = 0.0
                end
            end
        end
    end
end

# ==============================================================================
# 5. Visualisation
# ==============================================================================

using Plots
plotlyjs()

println("✅ Simulation terminée.")

# Espèces à afficher : protéines uniquement (lisibilité)
proteins     = [:D, :M, :I, :C, :E_int, :T, :E_ext]
proteins_idx = [species_index[s] for s in proteins]

p1 = plot(title="Bactérie 1 — protéines")
p2 = plot(title="Bactérie 2 — protéines (mémoire)")
p3 = plot(title="Bactérie 3 — protéines")
for (s, sym) in zip(proteins_idx, proteins)
    plot!(p1, history[:, 1, s], label=string(sym))
    plot!(p2, history[:, 2, s], label=string(sym))
    plot!(p3, history[:, 3, s], label=string(sym))
end

final_plot = plot(p1, p2, p3, layout=(3,1), size=(1400, 700))
#savefig(final_plot, "simulation_hebbian.pdf")
