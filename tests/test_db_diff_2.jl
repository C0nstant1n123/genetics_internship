# Import Packages 
include("../src/BioKan.jl")
using .BioKan
using Catalyst, JumpProcesses, Distributions, Plots

# Si ce fichier existe et contient la logique du circuit
include("../lib/rate-distortion-example/src/RateDistortionBiology.jl")
using .RateDistortionBiology

# ==============================================================================
# 1. PARAMÈTRES GLOBAUX
# ==============================================================================
dt = 0.1 # Pas de temps (1/10 c'est 0.1)
total_steps = Int(50000.0 / dt)
times = (1:total_steps) .* dt # On prépare le vecteur temps une fois pour toutes

species_names = [:X, :Y, :Y_trans, :Z, :Prom, :Prom_act, :mRNA] 
n_species = length(species_names)

n_bacteries = 1
taille_espace = 0.1
distance_comm = 0.1   
R_cell = 0.5e-6
# ==============================================================================
# 2. CONSTRUCTION DU CIRCUIT
# ==============================================================================

net = BioNetwork(distance_comm, 5) 

# On suppose que cette fonction renvoie le ReactionSystem et les paramètres par défaut
circuit_input, p_defaults_vec_input = create_genetic_simple_circuit_input_integrated(:node_generic)

# Création du dictionnaire de paramètres
params_dict_input = Dict(parameters(circuit_input) .=> p_defaults_vec_input)

# Modification des constantes spécifiques
# Note : Assure-toi que les clés sont bien des Symboles ou des objets Parameter
set_param!(params_dict_input, :k_create_X, 0.029)
set_param!(params_dict_input, :k_deg_X, 2.9e-5)

# ==============================================================================
# 3. INITIALISATION DE LA BACTÉRIE
# ==============================================================================
# État initial (Concentrations / Nombres de molécules)
u0_dict_raw_source = Dict(
    :X => 30.0, 
    :Y => 0.0, 
    :Y_trans => 0.0, 
    :Z => 0.0, 
    :Prom => 1.0, 
    :Prom_act => 0.0, 
    :mRNA => 0.0
)

# On mappe les symboles vers les espèces Catalyst si nécessaire
# (Si ton Bacterium gère ça en interne, tu peux passer u0_dict_raw_source directement)
# u0_source_dict = map_symbols_to_species(circuit_input, u0_dict_raw_source)

# Création de l'agent
# Hypothèse : Le constructeur Bacterium initialise le JumpProblem
b1 = Bacterium(1, [0.003100, 0.005], circuit_input, params_dict_input, u0_dict_raw_source)


add_bacterium!(net, b1)


# Build edges
build_edges!(net)
println("Réseau : $(length(net.nodes)) bactéries.")
println("Distance B1->B2 : $(net.edges[1])")




# D in m²/min : Diffuion Parrameters
D_dict = Dict(
    :X => 0.0,     
    :Y => 0.0,
    :Y_trans => 2.4e-8,
    :Z => 0.0,  
    :Prom => 0.0,  
    :Prom_act => 0.0, 
    :mRNA => 0.0   
)

# Gamma in 1/min : Degradation Parrameters
gamma_dict = Dict(
    :X => 2.9e-3,    
    :Y => 2.9e-3, 
    :Y_trans => 0.0,   
    :Z => 2.9e-3,    
    :Prom => 0.0,  
    :Prom_act => 0.0, 
    :mRNA => 0.0

)

# Build kernel : Computes K(r, t) for each edge and each species, returns a Dict with keys (id_s, id_t) and values K_edge (matrix n_species x n_steps)

# --- INITIALISATION AVANT BOUCLE ---
# 1. Calcul des poids (Une seule fois !)
weights_matrix = compute_static_coupling_physics(net.edges, D_dict, gamma_dict, species_names, R_cell, dt)


# 2. Tableaux temporaires pour éviter les allocations mémoire à chaque tour
flux_emissions = zeros(Float64, n_bacteries, n_species)
retained_stock = zeros(Float64, n_bacteries, n_species)
# ==============================================================================
# 4. BOUCLE DE SIMULATION
# ==============================================================================


println("Démarrage de la simulation...")

history_all = zeros(total_steps, n_bacteries, n_species)

# --- BOUCLE TEMPORELLE ---
for step in 1:total_steps
    t_sim = step * dt
    
    # === A. BIOLOGIE INTERNE (Réactions) ===
    for (id, b) in net.nodes
        # Met à jour les réactions internes (Hill, Michaelis-Menten, etc.)
        step_bacterium!(b, dt) 
    end

    # === B. GESTION DES FLUX (Qui part ? Qui reste ?) ===
    # Reset des buffers
    fill!(flux_emissions, 0.0)
    fill!(retained_stock, 0.0)

    for i in 1:n_bacteries
        if haskey(net.nodes, i)
            b = net.nodes[i] 
            
            for s in 1:n_species
                # Quantité totale actuelle dans la bactérie
                qty = max(0.0, get_species(b, species_names[s]))
                D_val = D_dict[species_names[s]]
                
                if D_val <= 1e-40
                    # Espèce FIXE : Tout reste dedans
                    retained_stock[i, s] = qty
                    flux_emissions[i, s] = 0.0
                else
                    # Espèce DIFFUSIVE : Fuite selon le coefficient de rétention
                    # retention_factor doit être défini (ex: 0.9 pour garder 90% par tour)
                    # Si tu ne l'as pas, on peut dire exp(-k_leak * dt)
                    p_retain = 0.0 # Exemple : 50% fuit à chaque seconde (très perméable)
                    
                    retained = rand(Binomial(ceil(Int, qty), p_retain))
                    leaked = qty - retained
                    
                    retained_stock[i, s] = retained
                    flux_emissions[i, s] = leaked
                end
            end
        end
    end

    # === C. COMMUNICATION (Transport Instantané) ===
    # C'est ici que la magie opère : flux_emissions est distribué aux voisins
    received_signals = propagate_signals_instantaneous!(weights_matrix, flux_emissions, n_bacteries, n_species)

    # === D. MISE À JOUR DE L'ÉTAT ===
    for i in 1:n_bacteries
        if haskey(net.nodes, i)
            b = net.nodes[i] 
            for s in 1:n_species
                sym = species_names[s]
                gamma = gamma_dict[sym]
                
                # Bilan de masse :
                # Nouveau = Ce qu'on a gardé + Ce qu'on a reçu des autres
                # (Note: Ce qu'on a émis est parti dans 'flux_emissions' et n'est pas réajouté)
                total_new = retained_stock[i, s] + (received_signals[i, s]*exp(-gamma * dt))
                

                
                

                total_new = ceil(Int, total_new)


                # Mise à jour dans la structure de données de la bactérie
                # On calcule le delta pour ta fonction add_input! ou on set directement
                current_val = get_species(b, sym) # Devrait être égal à qty calculé plus haut
                
                # ATTENTION : step_bacterium a peut-être déjà modifié des choses, 
                # mais ici on écrase le stock de particules par le résultat du transport.
                set_species!(b, sym, Float64(total_new))
                
                # Logging
                history_all[step, i, s] = total_new
            end
        end
    end

    if step % 1000 == 0
        println("Step $step complété.")
    end

end # Fin boucle

println("Simulation terminée.")

# ==============================================================================
# 5. VISUALISATION (CORRIGÉE)
# ==============================================================================

# Option 1 : Plotter seulement X (Indice 1)
p1 = plot(times, history_all[:, 1,1],   
    title  = "Bactérie 1 (Source) - Espèce X",
    label  = "X",
    color  = :blue,
    lw     = 2,
    xlabel = "Temps (s)",
    ylabel = "Molécules"
)

# Option 2 : Plotter TOUT pour vérifier la dynamique
# permutedims transforme le vecteur colonne des noms en ligne pour la légende
p2 = plot(times, history_all[:,1,:],
    title  = "Dynamique complète",
    label = permutedims(string.(species_names)), 
    lw     = 2,
    xlabel = "Temps (s)",
    layout = (n_species, 1), # Un graphe par espèce pour y voir clair
    size   = (800, 1000),
    legend = :outertopright
)

# Afficher le plot principal (p1 ou p2)
display(p1)
# display(p2) # Décommenter pour voir tout
