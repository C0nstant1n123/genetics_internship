module BioSim

# On suppose que BioKan est chargé dans le contexte global via le runner
using Catalyst, JumpProcesses, Distributions, LinearAlgebra
# Si tu lances en local pour tester, assure-toi de faire "using .BioKan" avant
import ..BioKan # On importe pour accéder aux fonctions utilitaires comme map_symbols_to_species

export run_single_simulation

# ==============================================================================
# FONCTION PRINCIPALE : MOTEUR DE SIMULATION
# ==============================================================================
function run_single_simulation(config::Dict)

    # --- 1. DÉBALLAGE DE LA CONFIGURATION ---
    # On récupère les blocs préparés par generate_configs.jl
    sim   = config[:sim]
    topo  = config[:topology]
    env   = config[:env]
    bio   = config[:biology] # C'est ici que tout est stocké maintenant

    # Paramètres temporels
    dt = sim[:dt]
    total_steps = Int(sim[:T_total] / dt) # On recalcule pour être sûr
    save_every = sim[:save_every]

    # Paramètres spatiaux
    n_bacteries = topo[:n_bacteries]
    R_cell = topo[:R_cell]
    dist_comm = topo[:distance_comm]

    # Noms des espèces (pour les boucles)
    species_names = bio[:species_names]
    n_species = length(species_names)

    # --- 2. CONSTRUCTION DU RÉSEAU ---
    net = BioKan.BioNetwork(dist_comm, n_bacteries)

    # A. Récupération des Circuits PRÉ-CONSTROITS
    # Plus besoin de create_genetic_... ici !
    rn_source = bio[:circuits][:source]
    rn_receiv = bio[:circuits][:receiver]

    # B. Récupération des Paramètres & U0 PRÉ-CALCULÉS
    params_source = bio[:parameters][:source]
    params_receiv = bio[:parameters][:receiver]
    
    # Mapping des symboles (:X) vers les objets Catalyst (@species X)
    # Cette étape est nécessaire car JLD2 sauvegarde des Dictionnaires de Symboles
    u0_source = BioKan.map_symbols_to_species(rn_source, bio[:u0][:source])
    u0_receiv = BioKan.map_symbols_to_species(rn_receiv, bio[:u0][:receiver])

    # C. Instanciation des Bactéries
    # On place B1 à 0.0 et B2 à la distance définie
    pos_b1 = topo[:pos_input]
    pos_b2 = topo[:pos_output]

    b1 = BioKan.Bacterium(1, pos_b1, rn_source, params_source, u0_source)
    b2 = BioKan.Bacterium(2, pos_b2, rn_receiv, params_receiv, u0_receiv)

    BioKan.add_bacterium!(net, b1)
    BioKan.add_bacterium!(net, b2)

    BioKan.build_edges!(net)

    # --- 3. PRÉPARATION PHYSIQUE ---
    
    # Récupération des coefficients (D et Gamma)
    # Note : Dans generate_configs, on les a appelés D_defaults et gamma_defaults
    D_dict = env[:D_defaults]
    gamma_dict = env[:gamma_defaults]
    Diff_targets_diff = env[:Diff_targets]

    # Calcul de la matrice de couplage (Pre-compute)
    weights_matrix = BioKan.compute_static_coupling_physics(net.edges, D_dict, gamma_dict, species_names, R_cell, dt)

    # --- 4. ALLOCATION MÉMOIRE (Buffers) ---
    flux_emissions = zeros(Float64, n_bacteries, n_species)
    retained_stock = zeros(Float64, n_bacteries, n_species)

    # Historique
    n_records = div(total_steps, save_every)
    history = zeros(Float32, n_records, n_bacteries, n_species)
    time_vector = zeros(Float64, n_records)

    # --- 5. BOUCLE DE SIMULATION ---
    
    save_idx = 1
    
    for step in 1:total_steps
        t_sim = step * dt

        # === A. BIOLOGIE INTERNE (Gillespie / ODE) ===
        for (_, b) in net.nodes
            BioKan.step_bacterium!(b, dt)
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

                    # Ce qu'on a gardé + Ce qui arrive (atténué par le voyage)
                    # Note: L'atténuation exponentielle est souvent intégrée dans weights_matrix 
                    # mais ici on l'applique explicitement pour la clarté si gamma > 0
                    
                    amount_received = received_signals[i, s]
                    
                    # Mise à jour
                    # Si gamma est fort, le signal reçu diminue encore pendant le dt
                    new_val = retained_stock[i, s] + (amount_received * exp(-gamma * dt))
                    
                    final_qty = floor(Int, max(0.0, new_val))
                    BioKan.set_species!(b, sym, Float64(final_qty))

                    # === E. SAUVEGARDE ===
                    if step % save_every == 0 && save_idx <= n_records
                        history[save_idx, i, s] = final_qty
                    end
                end
            end
        end
        
        if step % save_every == 0 && save_idx <= n_records
            time_vector[save_idx] = t_sim
            save_idx += 1
        end

    end # Fin boucle temps

    return history, time_vector
end

end # Fin Module
