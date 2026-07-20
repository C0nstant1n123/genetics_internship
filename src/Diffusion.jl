"""

    Diffusion

The aim of this module is to privide da Quorum sensing model of molecule between bacterias.
The whole system is caracterised by two main caracteristic times :

- The time of chemical integration of the genetic circuit : tau_r, 
which can be chosen as the carcteristic time of the slowest reaction 
of the system at a first time.  
- The time of diffusion : tau_d, 
ie the mean time for a molecule being outside the molecules

If those two carteric values are similar, the two mechanisms come together : 
 we need to know the passed states of all the bacterias and integrate it as a kernel. 

If tau_d >> tau_e (witch may be the case as some calculus of order of magnitude show it) ,
 we can do a big step of diffusion by only knowing the last states of the bacterias


"""




module Diffusion 
    using Distributions
    using ..BioNetworks

    export compute_diffusion_kernels_physics, update_diffusion!, compute_static_coupling_physics, propagate_signals_instantaneous!, compute_fpt_kernels, propagate_signals_delayed!








    function compute_diffusion_kernels_physics(edges, D_dict, gamma_dict, species_list, dt, T_max, R)
        t_vec = collect(dt:dt:T_max)
        n_steps = length(t_vec)
        n_species = length(species_list)
        kernels = Dict{Tuple{Int, Int}, Matrix{Float64}}()

        for (id_s, id_t, r) in edges
            K_edge = zeros(Float64, n_species, n_steps)

            for (s_idx, s_name) in enumerate(species_list)
                D = D_dict[s_name]
                γ = gamma_dict[s_name]

                if D <= 1e-40
                    if id_s == id_t 
                        @. K_edge[s_idx, :] = 1.    

                    else
                        @. K_edge[s_idx, :] = 0.0
                    end
                

                else 
                    term_diff = R^2 ./ (4.0 * pi * D * t_vec .+ R^2) .* exp.(-(r^2) ./ (4.0 * D * t_vec .+ R^2))
                    
                    
                    if id_s == id_t
                        @. K_edge[s_idx, :] = term_diff 
                        println("Self-loop (Bactérie $id_s, Espèce $s_name) : Retention initiale = $(K_edge[s_idx, 1])") 
                    else
                        @. K_edge[s_idx, :] =  term_diff .* exp.(-γ * t_vec)
                    end
                end
            end
            kernels[(id_s, id_t)] = K_edge
        end
        return kernels, t_vec, n_species
    end



    function update_diffusion!(buffer, kernels, new_emissions, current_ptr)
        n_bac, n_spec, n_steps = size(buffer)


        buffer[:, :, current_ptr] .= new_emissions

        received_signals = zeros(n_bac, n_spec)

        for ((id_s, id_t), K) in kernels
            if id_s != id_t 
                for s in 1:n_spec
                    sig_sum = 0.0
                    for τ in 1:n_steps
                        past_idx = mod1(current_ptr - τ + 1, n_steps)
                        sig_sum += max(0.0, rand(Binomial(round(Int, buffer[id_s, s, past_idx]), K[s, τ]))) 
                    end
                    received_signals[id_t, s] += sig_sum
                end
            end
        end

        next_ptr = mod1(current_ptr + 1, n_steps)
        return received_signals, next_ptr
    end






        """
    Génère les poids de couplage statiques (Probabilité d'atteinte par mouvement Brownien 3D).
    """
    function compute_static_coupling_physics(edges, D_dict, Gamma_dict, species_list, R_cell, dt)
        n_species = length(species_list)
        weights = Dict{Tuple{Int, Int}, Vector{Float64}}()

        for (id_s, id_t, dist_r) in edges
            W_edge = zeros(Float64, n_species)

            for (s_idx, s_name) in enumerate(species_list)
                D = D_dict[s_name]
                Gamma = Gamma_dict[s_name]

                # Si l'espèce ne diffuse pas, ou si on regarde la bactérie elle-même
                if D <= 1e-40 || id_s == id_t
                    W_edge[s_idx] = 0.0
                else
                    # Distance centre-à-centre (minorée pour éviter les divisions par zéro)
                    eff_dist = max(dist_r, 2.0 * R_cell) 
                    
                    # Longueur de diffusion (distance moyenne parcourue avant dégradation)
                    # Si Gamma est nul, la molécule ne se dégrade jamais (lambda infini)
                    lambda_diff = (Gamma > 0.0) ? sqrt(D / Gamma) : Inf
                    
                    # PROBABILITÉ BROWNIENNE 3D d'une molécule (Loi en 1/r + dégradation spatiale)
                    # La probabilité de heurter une sphère de rayon R_cell à une distance eff_dist
                    p_hit = (R_cell / eff_dist) * exp(-(eff_dist - R_cell) / lambda_diff)
                    
                    # On s'assure que la probabilité reste mathématiquement valide[0, 1]
                    W_edge[s_idx] = clamp(p_hit, 0.0, 1.0)
                end
            end
            
            # On ne stocke l'arête que s'il y a au moins une molécule qui peut diffuser
            if sum(W_edge) > 0.0
                weights[(id_s, id_t)] = W_edge
            end
        end
        return weights
    end

    """
    Propage les signaux instantanément (Tirage stochastique exact pour N discret).
    Version in-place : remplit `received_signals` (buffer pré-alloué), évite toute allocation.
    """
    function propagate_signals_instantaneous!(received_signals::Matrix{Float64},
                                              weights, flux_emissions,
                                              _n_bacteries, n_species)
        fill!(received_signals, 0.0)

        for ((id_s, id_t), W_vec) in weights
            for s in 1:n_species
                w = W_vec[s]
                if w > 0.0
                    amount_leaving = flux_emissions[id_s, s]
                    if amount_leaving > 0.0
                        n_molecules = ceil(Int, amount_leaving)
                        received_signals[id_t, s] += rand(Binomial(n_molecules, w))
                    end
                end
            end
        end

        return received_signals
    end

    """
    Compatibilité : version sans buffer — alloue une matrice (utiliser de préférence la version in-place).
    """
    function propagate_signals_instantaneous!(weights, flux_emissions, n_bacteries, n_species)
        received_signals = zeros(Float64, n_bacteries, n_species)
        return propagate_signals_instantaneous!(received_signals, weights, flux_emissions,
                                                n_bacteries, n_species)
    end


    """
        propagate_signals_delayed!(received, weights_delayed, weights_instant,
                                   flux_emissions, delay_buffer, ptr, delay_steps, n_species)

    Transport avec délai fixe pour certaines espèces, instantané pour les autres.

    - `weights_delayed` : poids pour les espèces avec délai (D_diff, E_ext_diff, C_diff)
    - `weights_instant` : poids pour les espèces instantanées (O_diff)
    - `delay_buffer`    : Array (n_bac, n_species, delay_steps) — buffer circulaire
    - `ptr`             : pointeur courant dans le buffer (Int)
    - `delay_steps`     : nombre de pas de délai

    Retourne (received, new_ptr).
    """
    function propagate_signals_delayed!(received::Matrix{Float64},
                                        weights_delayed, weights_instant,
                                        flux_emissions::Matrix{Float64},
                                        delay_buffer::Array{Float64,3},
                                        ptr::Int, delay_steps::Int, n_species::Int)
        fill!(received, 0.0)

        # Lire d'abord le slot courant (contient ce qui a été écrit il y a delay_steps pas)
        # puis écraser avec les émissions d'aujourd'hui → délai exact de delay_steps pas
        for ((id_s, id_t), W_vec) in weights_delayed
            for s in 1:n_species
                w = W_vec[s]
                if w > 0.0
                    amount = delay_buffer[id_s, s, ptr]
                    if amount > 0.0
                        n_mol = ceil(Int, amount)
                        received[id_t, s] += rand(Binomial(n_mol, w))
                    end
                end
            end
        end

        # Écrire les émissions courantes dans le slot (écrase l'ancien)
        delay_buffer[:, :, ptr] .= flux_emissions

        # Propager instantanément les espèces sans délai (O_diff)
        for ((id_s, id_t), W_vec) in weights_instant
            for s in 1:n_species
                w = W_vec[s]
                if w > 0.0
                    amount = flux_emissions[id_s, s]
                    if amount > 0.0
                        n_mol = ceil(Int, amount)
                        received[id_t, s] += rand(Binomial(n_mol, w))
                    end
                end
            end
        end

        new_ptr = mod1(ptr + 1, delay_steps)
        return received, new_ptr
    end


    """
        compute_fpt_kernels(edges, D_dict, gamma_dict, species_list, dt, R)

    Calcule le noyau de premier passage (FPT) pour chaque arête du réseau.

    Pour une molécule émise depuis id_s et devant atteindre id_t (rayon R, distance r),
    la densité de probabilité d'arriver pour la **première fois** au temps τ est :

        K(τ) = (R/r) · (d/√(4πD)) · τ^(-3/2) · exp(−d²/(4Dτ) − γτ)

    où d = r − R est la distance surface-à-surface.

    L'intégrale sur τ ≥ 0 vaut exactement p_hit = (R/r)·exp(−d/λ), λ=√(D/γ),
    ce qui est identique à `compute_static_coupling_physics` — les deux méthodes sont cohérentes.

    Le noyau K[s, τ] retourné est **multiplié par dt** : c'est la probabilité qu'une molécule
    émise au temps 0 arrive pendant [τ·dt, (τ+1)·dt]. Il est utilisé directement comme
    paramètre p du tirage Binomial dans `update_diffusion!`.

    T_max adaptatif : 5/γ_min (parmi les espèces diffusantes), en secondes.
    """
    function compute_fpt_kernels(edges, D_dict, gamma_dict, species_list, dt, R; eps_cut=1e-2, T_max_abs=500.0)
        n_species = length(species_list)

        # Passe 1 : taille du buffer = τ_peak + n_tau_widths / γ
        # Après τ_peak la queue décroît comme exp(-γτ).
        # n_tau_widths = nombre de demi-vies qu'on garde (défaut 3 → e^{-3}≈5% du pic).
        # On prend le pire cas sur toutes les arêtes et espèces diffusantes.
        n_tau_widths = -log(eps_cut)   # eps_cut=1e-2 → 4.6 demi-vies, 1e-3 → 6.9, etc.
        n_steps = 1
        for (_, _, r) in edges
            for s_name in species_list
                D = D_dict[s_name]; γ = gamma_dict[s_name]
                (D <= 1e-40 || γ <= 1e-40) && continue
                λ = sqrt(D / γ)
                r_eff = max(r, R + 1e-12)
                d = max(r_eff - R, 1e-12)
                τ_peak = (-3.0 + sqrt(9.0 + (2.0*d/λ)^2)) / (4.0 * γ)
                τ_cut  = τ_peak + n_tau_widths / γ
                n_steps = max(n_steps, round(Int, τ_cut / dt) + 1)
            end
        end
        n_steps = min(n_steps, round(Int, T_max_abs / dt))
        t_vec = collect(dt:dt:(n_steps * dt))

        kernels = Dict{Tuple{Int,Int}, Matrix{Float64}}()

        for (id_s, id_t, r) in edges
            K_edge = zeros(Float64, n_species, n_steps)

            for (s_idx, s_name) in enumerate(species_list)
                D = D_dict[s_name]
                γ = gamma_dict[s_name]

                if D <= 1e-40
                    # Espèce non-diffusante : self-loop = retention totale
                    id_s == id_t && (K_edge[s_idx, 1] = 1.0)
                elseif id_s == id_t
                    # Auto-boucle : retention instantanée (pas de transit)
                    K_edge[s_idx, 1] = 1.0
                else
                    r_eff = max(r, R + 1e-12)
                    d     = max(r_eff - R, 1e-12)   # distance surface-à-surface

                    # Noyau FPT : (R/r)·(d/√(4πD))·τ^(-3/2)·exp(−d²/(4Dτ) − γτ)
                    # Multiplié par dt → probabilité par pas de temps
                    @. K_edge[s_idx, :] = (R / r_eff) * (d / sqrt(4π * D)) *
                        t_vec^(-1.5) * exp(-d^2 / (4.0 * D * t_vec) - γ * t_vec) * dt

                    # Sécurité numérique (ne doit jamais dépasser 1)
                    clamp!(view(K_edge, s_idx, :), 0.0, 1.0)
                end
            end

            kernels[(id_s, id_t)] = K_edge
        end

        return kernels, t_vec, n_steps
    end

end