# Import Packages 
include("../src/BioKan.jl")
using .BioKan
using Catalyst, JumpProcesses, Distributions, Plots
using Distributions
include("../lib/rate-distortion-example/src/RateDistortionBiology.jl")
using .RateDistortionBiology


# 0. Input Functions _______________________________________

function periodic_step_input_signal(t, high_val=100.0, low_val=0.0; period=1000.0, duty_cycle=0.5)
    cycle_pos = t % period
    
    if cycle_pos < (period * duty_cycle)
        return high_val
    else
        return low_val
    end
end

function next_poisson_signal(current_val, high_val, low_val, rate, dt)
    p_switch = rate * dt
    
    if rand() < p_switch
        if isapprox(current_val, high_val, atol=1e-5)
            return low_val
        else
            return high_val
        end
    else
        return current_val
    end

end

# 1. General parmeters _________________________________________________________

n_bacteries = 2
taille_espace = 0.1
distance_comm = 0.1   
dt = 1/10
total_steps = Int(200000.0 / dt)
R_cell = 0.5e-6
input_node_id = 1 


# Rapport fixe entre les deux constantes
k_degradation_X = 2.9e-3  

# Valeurs de k_create espacées logarithmiquement (de 0.001 à 1.0)
n_points = 10
ratios = range(0.01,5,n_points)
println(ratios)

# Génère les couples en respectant le rapport
param_pairs = [(k_degradation_X*ratio, k_degradation_X) for ratio in ratios]

DATA = []



for (k_c, k_d) in param_pairs

    # 2. Build Network ____________________________________________________________

    net = BioNetwork(distance_comm, 5) 

    # Build circuits
    species_names = [:X, :Y, :Y_trans, :Z, :Prom, :Prom_act, :mRNA] 
    n_species = length(species_names) 

    circuit_input, p_defaults_vec_input = create_genetic_simple_circuit_input_integrated(:node_generic)   # could be create_genetic_simple_circuit_input_integrated
    circuit_output, p_defaults_vec_output = create_genetic_simple_circuit_output(:node_generic)


    # Add Circuits Parrameters
    params_dict_input = Dict(parameters(circuit_input) .=> p_defaults_vec_input)
    params_dict_output = Dict(parameters(circuit_output) .=> p_defaults_vec_output)

    



    set_param!(params_dict_input, :k_create_X, k_c)
    set_param!(params_dict_input, :k_deg_X, k_d)

    println(params_dict_input)

    #  Build Bacteries
    u0_dict_raw_source = Dict(:X => k_c/k_d, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :Prom => 1.0, :Prom_act => 0.0, :mRNA => 0.0)
    u0_dict_raw_receiv = Dict(:X => 0.0, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :Prom => 1.0, :Prom_act => 0.0, :mRNA => 0.0)


    u0_source_dict = map_symbols_to_species(circuit_input, u0_dict_raw_source)
    u0_receiv_dict = map_symbols_to_species(circuit_output, u0_dict_raw_receiv)

    b1 = Bacterium(1, [0.003100, 0.005], circuit_input, params_dict_input, u0_source_dict)
    b2 = Bacterium(2, [0.003103, 0.005], circuit_output, params_dict_output, u0_receiv_dict)

    add_bacterium!(net, b1)
    add_bacterium!(net, b2)

    # Build edges
    build_edges!(net)
    println("Réseau : $(length(net.nodes)) bactéries.")
    println("Distance B1->B2 : $(net.edges[2])")



    # 3. Diffusion Kernels __________________________________________________________



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

    println("Simulation : Démarrage (Mode Instantané)...")

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
    # 7. VISUALISATION
    # ==============================================================================
    times = (1:total_steps) .* dt
    colors = [:blue :red :green :cyan] 
    labels = ["X", "Y","Y_trans","Z"]

    p1 = plot(times, history_all[:, 1, 1:4], title="Bactérie 1 (Source)", label=labels, color=colors, lw=2 )   #, xlims= (1e4,9e4)

        
    p2 = plot(times, history_all[:, 2, 1:4], title="Bactérie 2 (Voisin)", label=labels, color=colors, lw=2)   #, xlims= (3e3,7e3)


    plot(p1, p2, layout=(1, 2), size=(900, 400))

    push!(DATA, history_all)

end



println(size(DATA))
println("maximum Y in Bactérie 2: ", maximum(DATA[1][:, 2, 2]))
println("maximum Y in Bactérie 1: ", maximum(history_all[:, 1, 2]))

println("maximum X in Bactérie 2: ", maximum(history_all[:, 2, 1]))
println("maximum X in Bactérie 1: ", maximum(history_all[:, 1, 1]))




# Ajouter Fenetre temporelle 

i=4
min_lag = 1    # Valeur minimale (ex: 1)
max_lag = 50000  # Valeur maximale (ex: 20, ajustez basé sur la taille de vos données)
step = 10

lags = min_lag:step:max_lag
mutual_infos = Float64[]  # Array typé pour stocker les MI (plus performant)
for lag in lags 
    len = size(DATA[i][:, 1, 1])[1]  # Taille en Int64
    if len < lag  # Vérifie si lag est trop grand (slice vide ou négatif)
        println("Lag $lag trop grand pour la taille des données ($len) – skipping")
        push!(mutual_infos, NaN)  # Ajoute NaN pour marquer les skips (optionnel)
        continue
    end
    X = DATA[i][1:(len - lag + 1), 1, 1]  # Premiers éléments de X
    Y = DATA[i][lag:len, 2, 4]            # Éléments décalés de Y (même taille)
    mi = mutual_information_continuous(X, Y)  # Calcul
    push!(mutual_infos, mi)               # Ajoute au array
    if lag % 121 == 0
        println("Mutual Info pour lag $lag : $mi")  # Optionnel : debug
    end
end




mutual_info = mutual_information_continuous(DATA[i][:, 1, 1],DATA[1][:, 2, :4])
p4 = plot(lags .*dt, mutual_infos)

