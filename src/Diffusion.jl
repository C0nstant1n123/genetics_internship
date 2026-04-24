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

    export compute_diffusion_kernels_physics, update_diffusion!, compute_static_coupling_physics, propagate_signals_instantaneous!








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
    """
    function propagate_signals_instantaneous!(weights, flux_emissions, n_bacteries, n_species)
        received_signals = zeros(Float64, n_bacteries, n_species)

        for ((id_s, id_t), W_vec) in weights

            for s in 1:n_species
                w = W_vec[s] # 'w' est maintenant la probabilité d'atteinte P_hit
                if w > 0.0
                    amount_leaving = flux_emissions[id_s, s]

                    if amount_leaving > 0.0
                        # On s'assure d'avoir un nombre entier de molécules qui partent
                        n_molecules = ceil(Int, amount_leaving)
                        
                        # Tirage stochastique : chaque molécule a une chance 'w' d'arriver
                        received = rand(Binomial(n_molecules, w))

                        received_signals[id_t, s] += received
                    end
                end
            end
        end

        return received_signals
    end
end