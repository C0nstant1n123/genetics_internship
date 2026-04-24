module BioSim

# On suppose que BioKan est chargé dans le contexte global via le runner
using Catalyst, JumpProcesses, Distributions, LinearAlgebra, Statistics, Printf
# Si tu lances en local pour tester, assure-toi de faire "using .BioKan" avant
import ..BioKan # On importe pour accéder aux fonctions utilitaires comme map_symbols_to_species

export run_single_simulation, HebbianConfig, default_hebbian_config, run_hebbian_simulation, run_hebbian_simulation_full, run_hebbian_simulation_threaded

# Seuil de détection d'explosion numérique (max_density = 60, donc 1e6 est aberrant)
const MAX_CONC = 1e6
# Fréquence de vérification (toutes les N étapes pour ne pas pénaliser le temps de simu)
const CHECK_EVERY = 100

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

    # A. Reconstruction locale des ReactionSystems (à partir du type stocké dans le config)
    # Le ReactionSystem n'est JAMAIS sérialisé/transporté : il est recréé ici, sur le worker.
    function build_circuit(type::Symbol)
        if type == :hill_repeater_input
            return BioKan.create_genetic_hill_repeter_input(:source)
        elseif type == :hill_repeater_output
            return BioKan.create_genetic_hill_repeter_output(:receiver)
        elseif type == :burst
            return BioKan.create_burst_circuit(:node)
        elseif type == :hebbian
            return BioKan.create_hebbian_model(:node)
        else
            error("Circuit type inconnu : $type")
        end
    end

    rn_source, _ = build_circuit(bio[:circuit_types][:source])
    rn_receiv, _ = build_circuit(bio[:circuit_types][:receiver])

    # B. Reconstruction des Dicts de paramètres depuis les valeurs numériques
    # L'ordre de parameters(rn) est déterministe et identique à celui utilisé dans generate_configs
    params_source = Dict(parameters(rn_source) .=> bio[:param_values][:source])
    params_receiv = Dict(parameters(rn_receiv) .=> bio[:param_values][:receiver])

    # C. Mapping u0
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
    # Diff_targets non utilisé ici (géré côté test scripts)

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


# ==============================================================================
# CONFIGURATION HEBBIAN
# ==============================================================================

struct HebbianConfig
    n_bacteries       :: Int
    d_max             :: Float64       # distance max de communication
    n_segments        :: Int           # spacing = d_max / n_segments
    dt                :: Float64
    R_cell            :: Float64
    species_names     :: Vector{Symbol}
    D_dict            :: Dict{Symbol, Float64}
    gamma_dict        :: Dict{Symbol, Float64}
    diffusion_targets :: Dict{Symbol, Symbol}
    max_density       :: Float64       # amplitude des spikes injectés (Input + teacher forcing)
end

function default_hebbian_config()
    species_names = [
        :D, :M, :I, :C, :E_int, :T, :E_ext,
        :D_diff, :C_diff, :E_ext_diff,
        :mRNA_D, :mRNA_M, :mRNA_I, :mRNA_C, :mRNA_T, :mRNA_E_ext
    ]

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
        :D => :D, :M => :M, :I => :I, :C => :C,
        :E_int => :E_int, :T => :T, :E_ext => :E_ext,
        :D_diff => :D, :C_diff => :C, :E_ext_diff => :E_ext,
        :mRNA_D => :mRNA_D, :mRNA_M => :mRNA_M, :mRNA_I => :mRNA_I,
        :mRNA_C => :mRNA_C, :mRNA_T => :mRNA_T, :mRNA_E_ext => :mRNA_E_ext
    )

    return HebbianConfig(27, 3.5e-5, 3, 0.1, 0.5e-6,
                         species_names, D_dict, gamma_dict, diffusion_targets,
                         60.0)   # max_density : amplitude des spikes (défaut = 60.0)
end

# ==============================================================================
# MOTEUR DE SIMULATION HEBBIAN
# ==============================================================================

function run_hebbian_simulation_full(θ::Vector{Float64},
                                     gate_name::Symbol,
                                     cfg::HebbianConfig)

    n_bacteries       = cfg.n_bacteries
    dt                = cfg.dt
    R_cell            = cfg.R_cell
    species_names     = cfg.species_names
    n_species         = length(species_names)
    D_dict            = cfg.D_dict
    gamma_dict        = cfg.gamma_dict
    diffusion_targets = cfg.diffusion_targets

    # --- 1. Pattern d'apprentissage ---
    time_steps, Input_A, Input_B, Output_0, Output_1, mask =
        BioKan.pattern_to_learn_density(gate_name; max_density = cfg.max_density)
    total_steps = length(time_steps)

    # --- 2. Circuit et paramètres ---
    circuit, _ = BioKan.create_hebbian_model(:hebbian_node)
    params_dict = Dict(parameters(circuit) .=> θ)
    u0_raw  = Dict(s => 0.0 for s in species_names)
    u0_dict = BioKan.map_symbols_to_species(circuit, u0_raw)

    # --- 3. Réseau cubique ---
    net = BioKan.BioNetwork(cfg.d_max, n_bacteries)
    BioKan.build_network_cube!(net, n_bacteries, cfg.d_max, cfg.n_segments,
                               circuit, params_dict, u0_dict, :ssa)
    node_roles = BioKan.assign_tetrahedral_roles(net)

    # --- 4. Matrice de couplage (précalculée) ---
    weights = BioKan.compute_static_coupling_physics(
        net.edges, D_dict, gamma_dict, species_names, R_cell, dt)

    # --- 5. Buffers ---
    # GC explicite avant les grosses allocations : force la collecte des
    # intégrateurs SSA et du history de l'appel précédent (le GC de Julia
    # est lazy et n'anticipe pas les pics sur un cluster sous pression RAM).
    GC.gc()
    flux_emissions = zeros(Float64, n_bacteries, n_species)
    history        = zeros(Float32, total_steps, n_bacteries, n_species)

    # --- 6. Boucle de simulation ---
    exploded = false
    for step in 1:total_steps

        # A. INPUTS (pattern structuré — teacher forcing inclus)
        BioKan.inject_pattern_step!(net, step,
                                    Input_A, Input_B, Output_0, Output_1,
                                    node_roles)

        # B. BIOLOGIE INTERNE
        for (_, b) in net.nodes
            BioKan.step_bacterium!(b, dt)
        end

        # C. FLUX
        fill!(flux_emissions, 0.0)
        for i in 1:n_bacteries
            haskey(net.nodes, i) || continue
            b = net.nodes[i]
            for s in 1:n_species
                sym = species_names[s]
                D_dict[sym] > 1e-40 &&
                    (flux_emissions[i, s] = max(0.0, BioKan.get_species(b, sym)))
            end
        end

        # D. TRANSPORT
        received = BioKan.propagate_signals_instantaneous!(
            weights, flux_emissions, n_bacteries, n_species)

        # E. UPDATE
        for i in 1:n_bacteries
            haskey(net.nodes, i) || continue
            b = net.nodes[i]
            for s in 1:n_species
                sym = species_names[s]
                if D_dict[sym] > 1e-40
                    gamma  = gamma_dict[sym]
                    target = diffusion_targets[sym]
                    amt    = max(0.0, received[i, s] * exp(-gamma * dt))
                    BioKan.set_species!(b, target, BioKan.get_species(b, target) + amt)
                    BioKan.set_species!(b, sym, 0.0)
                end
                history[step, i, s] = max(0.0f0, Float32(BioKan.get_species(b, sym)))
            end
        end

        # F. VÉRIFICATION EXPLOSION — toutes les CHECK_EVERY étapes
        if step % CHECK_EVERY == 0 && maximum(@view history[step, :, :]) > MAX_CONC
            exploded = true
            val, ci = findmax(@view history[step, :, :])
            bact_id, sp_id = ci[1], ci[2]
            t_exp = time_steps[step]
            @printf "  [EXPLOSION] t=%.1f (step %d/%d) | bact=%d | espèce=%s | valeur=%.2e\n" t_exp step total_steps bact_id species_names[sp_id] val
            flush(stdout)
            break
        end
    end

    # --- 7. Loss ---
    exploded && return Inf, history, time_steps, node_roles, Input_A, Input_B, Output_0, Output_1, mask, net

    # K_DM = θ[12] : seuil d'intégration D→M, utilisé pour normaliser D dans la loss
    d_idx = findfirst(==(:D), species_names)
    loss  = BioKan.compute_loss(history, time_steps, mask, node_roles, d_idx)
    return loss, history, time_steps, node_roles, Input_A, Input_B, Output_0, Output_1, mask, net
end

# Méthode courte pour CMA-ES — retourne uniquement la loss
# Utilise run_hebbian_simulation_full (sans Threads.@threads interne) pour éviter
# la contention quand CMA-ES évalue déjà les candidats en parallèle (:thread)
function run_hebbian_simulation(θ::Vector{Float64},
                                gate_name::Symbol,
                                cfg::HebbianConfig)::Float64
    loss, = run_hebbian_simulation_full(θ, gate_name, cfg)
    return loss
end

# ==============================================================================
# MOTEUR DE SIMULATION HEBBIAN — VERSION THREADÉE
# Identique à run_hebbian_simulation_full, étape B parallélisée via Threads.@threads
# ==============================================================================

function run_hebbian_simulation_threaded(θ::Vector{Float64},
                                         gate_name::Symbol,
                                         cfg::HebbianConfig)

    n_bacteries       = cfg.n_bacteries
    dt                = cfg.dt
    R_cell            = cfg.R_cell
    species_names     = cfg.species_names
    n_species         = length(species_names)
    D_dict            = cfg.D_dict
    gamma_dict        = cfg.gamma_dict
    diffusion_targets = cfg.diffusion_targets

    time_steps, Input_A, Input_B, Output_0, Output_1, mask =
        BioKan.pattern_to_learn_density(gate_name; max_density = cfg.max_density)
    total_steps = length(time_steps)

    circuit, _ = BioKan.create_hebbian_model(:hebbian_node)
    params_dict = Dict(parameters(circuit) .=> θ)
    u0_raw  = Dict(s => 0.0 for s in species_names)
    u0_dict = BioKan.map_symbols_to_species(circuit, u0_raw)

    net = BioKan.BioNetwork(cfg.d_max, n_bacteries)
    BioKan.build_network_cube!(net, n_bacteries, cfg.d_max, cfg.n_segments,
                               circuit, params_dict, u0_dict, :ssa)
    node_roles = BioKan.assign_tetrahedral_roles(net)

    weights = BioKan.compute_static_coupling_physics(
        net.edges, D_dict, gamma_dict, species_names, R_cell, dt)

    GC.gc()
    flux_emissions = zeros(Float64, n_bacteries, n_species)
    history        = zeros(Float32, total_steps, n_bacteries, n_species)
    ids            = collect(1:n_bacteries)

    # ── Pré-calculs AVANT la boucle ──────────────────────────────────
    valid_nodes  = [net.nodes[i] for i in 1:n_bacteries if haskey(net.nodes, i)]
    valid_ids    = [i            for i in 1:n_bacteries if haskey(net.nodes, i)]

    diffusible   = [D_dict[species_names[s]] > 1e-40       for s in 1:n_species]
    decay_vec    = [exp(-gamma_dict[species_names[s]] * dt) for s in 1:n_species]
    target_vec   = [diffusion_targets[species_names[s]]     for s in 1:n_species]

    received     = zeros(Float64, n_bacteries, n_species)   # buffer réutilisé

    # ── Boucle temporelle ─────────────────────────────────────────────
    exploded = false
    for step in 1:total_steps

        # A. INPUTS — inchangé
        BioKan.inject_pattern_step!(net, step, Input_A, Input_B, Output_0, Output_1, node_roles)

        # B. BIOLOGIE — plus de Dict lookup
        Threads.@threads for b in valid_nodes
            BioKan.step_bacterium!(b, dt)
        end

        # C. FLUX — diffusible[s] au lieu de D_dict[sym]
        fill!(flux_emissions, 0.0)
        for (k, b) in enumerate(valid_nodes)
            i = valid_ids[k]
            for s in 1:n_species
                diffusible[s] || continue
                v = BioKan.get_species(b, species_names[s])
                v > 0.0 && (flux_emissions[i, s] = v)
            end
        end

        # D. TRANSPORT — inchangé pour l'instant
        received = BioKan.propagate_signals_instantaneous!(
            weights, flux_emissions, n_bacteries, n_species)

        # E. UPDATE — decay_vec précalculé, plus de exp() dans la boucle
        for (k, b) in enumerate(valid_nodes)
            i = valid_ids[k]
            for s in 1:n_species
                if diffusible[s]
                    amt = max(0.0, received[i, s] * decay_vec[s])   # ← plus de exp()
                    BioKan.set_species!(b, target_vec[s],
                        BioKan.get_species(b, target_vec[s]) + amt)
                    BioKan.set_species!(b, species_names[s], 0.0)
                end
                history[step, i, s] = max(0.0f0, Float32(BioKan.get_species(b, species_names[s])))
            end
        end

        # F. VÉRIFICATION EXPLOSION — toutes les CHECK_EVERY étapes
        if step % CHECK_EVERY == 0 && maximum(@view history[step, :, :]) > MAX_CONC
            exploded = true
            break
        end
    end

    exploded && return Inf, history, time_steps, node_roles, Input_A, Input_B, Output_0, Output_1, mask, net

    d_idx = findfirst(==(:D), species_names)
    loss  = BioKan.compute_loss(history, time_steps, mask, node_roles, d_idx)
    return loss, history, time_steps, node_roles, Input_A, Input_B, Output_0, Output_1, mask, net
end

end # Fin Module
