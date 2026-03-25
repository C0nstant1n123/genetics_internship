# Import Packages 
include("../src/BioKan.jl")
using .BioKan
using Catalyst, JumpProcesses, Distributions, Plots
using Distributions

function generate_burst!(b::Bacterium, species::Symbol, val::Float64, prob::Float64)
    # Tirage aléatoire (Bernoulli)
    if rand() < prob
        # On force la valeur dans l'intégrateur
        add_input!(b, species, val)
        return true # Indique qu'un burst a eu lieu (utile pour le debug/plot)
    end
    return false
end

# 1. General parmeters _________________________________________________________

n_bacteries = 3
taille_espace = 0.1
distance_comm = 0.1   
dt = 1/10
total_steps = Int(10000.0 / dt)
R_cell = 0.5e-6
input_node_id = [1,3] 

# 2. Build Network ____________________________________________________________

net = BioNetwork(distance_comm, n_bacteries ) 

species_names = [:X, :Y, :Z, :mRNA_X, :mRNA_Y, :mRNA_Z, :X_diff]
species_index = Dict(sym => s for (s, sym) in enumerate(species_names))
n_species = length(species_names) 

diffusion_targets = Dict(
    :X => :X,
    :Y => :Y,
    :Z => :Z,
    :mRNA_X => :mRNA_X,
    :mRNA_Y => :mRNA_Y,
    :mRNA_Z => :mRNA_Z,
    :X_diff => :X
)

circuit, p_defaults_vec = create_burst_circuit(:node_generic)
params_dict = Dict(parameters(circuit) .=> p_defaults_vec)


u0_dict_raw = Dict(:X => 0.0, :Y => 0.0, :Z => 0.0, :mRNA_X => 0.0, :mRNA_Y => 0.0, :mRNA_Z => 0.0, :X_diff => 0.0)
u0_dict  =map_symbols_to_species(circuit, u0_dict_raw)

D_dict = Dict(:X => 0.0, :Y => 0.0, :Z => 0.0, :mRNA_X => 0.0, :mRNA_Y => 0.0, :mRNA_Z => 0.0, :X_diff => 1.0e-9) #1.0e-9
gamma_dict = Dict(:X => 0.0, :Y => 0.0, :Z => 0.0, :mRNA_X => 0.0, :mRNA_Y => 0.0, :mRNA_Z => 0.0, :X_diff => 2.9e-2)

b1 = Bacterium(1, [0.003100, 0.005], circuit, params_dict, u0_dict; mode=:ssa)
b2 = Bacterium(2, [0.003103, 0.005], circuit, params_dict, u0_dict; mode=:ssa)
b3 = Bacterium(3, [0.003106, 0.005], circuit, params_dict, u0_dict; mode=:ssa)

add_bacterium!(net,b1)
add_bacterium!(net,b2)
add_bacterium!(net,b3)

build_edges!(net)
println("Réseau : $(length(net.nodes)) bactéries.")
println("Distance B1->B2 : $(net.edges[2])")




# ==============================================================================
# PARTIE SIMULATION & INTEGRATION
# ==============================================================================

# Initialisation de la matrice d'historique
# Dimensions : [Temps, ID_Bactérie, Espèce]
weights_matrix = BioKan.compute_static_coupling_physics(net.edges, D_dict, gamma_dict, species_names, R_cell, dt)


# Paramètres du Burst
burst_prob = 0.001 # 1% de chance par dt (ajuste selon la fréquence voulue)
burst_intensity = 30.0

# --- 4. ALLOCATION MÉMOIRE (Buffers) ---
flux_emissions = zeros(Float64, n_bacteries, n_species)
retained_stock = zeros(Float64, n_bacteries, n_species)

# Historique

history = zeros(total_steps, n_bacteries, n_species)


println("🚀 Démarrage de la simulation ($total_steps pas)...")

for step in 1:total_steps
    # Temps actuel
    t_sim = step * dt

    # === A. INPUT (BURST) ===
    # On applique le burst uniquement sur le nœud d'entrée (b1 / id 1)
    for input_node_id in input_node_id 
        if haskey(net.nodes, input_node_id)
            input_bact = net.nodes[input_node_id]
            triggered = generate_burst!(input_bact, :X, burst_intensity, burst_prob)
            if triggered
                println("⚡ Burst sur X au temps $t_sim (nœud $input_node_id)")
            end
        end
    end


    # === B. BIOLOGIE INTERNE (Réactions) ===
    for (id, b) in net.nodes
        # 1. Évolution des équations différentielles locales
        step_bacterium!(b, dt) 
        if BioKan.get_species(b1, :X_diff ) > 0
            #println("oui")
        end
    end

            # === B. GESTION DES FLUX (Diffusion Instantanée) ===
    fill!(flux_emissions, 0.0)
    fill!(retained_stock, 0.0)

    for i in 1:n_bacteries
        if haskey(net.nodes, i)
            b = net.nodes[i]
            for s in 1:n_species
                sym = species_names[s]
                qty = max(0.0, BioKan.get_species(b, sym))
                
                D_val = get(D_dict, sym, 0.0)

                # LOGIQUE PHYSIQUE
                if D_val <= 1e-40
                    # Ne diffuse pas
                    retained_stock[i, s] = qty
                else
                    # Diffuse (Mode Instantané = Tout sort)
                    # Tu peux remettre le tirage Binomial ici si tu veux du bruit
                    retained = 0.0 
                    leaked = qty
                    
                    retained_stock[i, s] = retained
                    flux_emissions[i, s] = leaked
                end

            end
        end
    end

    # === C. TRANSPORT (Via Matrice de Poids) ===
    # Calcule qui reçoit quoi en fonction des distances
    received_signals = BioKan.propagate_signals_instantaneous!(weights_matrix, flux_emissions, n_bacteries, n_species)

    # === D. UPDATE & DÉGRADATION EXTERNE ===
    for i in 1:n_bacteries
        if haskey(net.nodes, i)
            b = net.nodes[i]
            for s in 1:n_species
                sym = species_names[s]
                gamma = get(gamma_dict, sym, 0.0)
                D_val = get(D_dict, sym, 0.0)

                if D_val <= 1e-40
                    # Non diffusible : step_bacterium! a déjà tout géré
                    # On lit juste la valeur pour la sauvegarder
                    history[step, i, s] = max(0.0, BioKan.get_species(b, sym))

                else
                    # Diffusible (X_diff) : on injecte dans la cible (X)
                    target_sym = diffusion_targets[sym]  # :X
                    target_idx = species_index[target_sym]

                    amount_received = received_signals[i, s] * exp(-gamma * dt)
                    received_int = max(0.0, amount_received)
                    if received_int > 0
                        #println("$i reçoit $received_int X au pas $step")
                    end

                    # On AJOUTE à X, sans écraser
                    current_X = BioKan.get_species(b, target_sym)
                    BioKan.set_species!(b, target_sym, current_X + received_int)

                    BioKan.set_species!(b, sym, 0.0)


                    # X_diff est vide dans la cellule
                    history[step, i, s] = 0.0

                end
            end
        end
    end
end # Fin boucle temps





using Plots
plotlyjs() # <--- On active le mode interactif

println("✅ Simulation terminée.")

p1 = plot(title="Bactérie 1")
p2 = plot(title="Bactérie 2")
p3 = plot(title="Bacterie 3")
for s in 1:n_species
    plot!(p1, history[:, 1, s], label=string(species_names[s]))
    plot!(p2, history[:, 2, s], label=string(species_names[s]))
    plot!(p3, history[:, 3, s], label=string(species_names[s]))
end
final_plot =plot(p1, p2, p3, layout=(3,1), size=(1400, 500))
#savefig(final_plot, "simulation.pdf")
