module Hebbian

    using ..Bacterias
    using ..BioNetworks
    using Random
    using Statistics
    using Printf

    export pattern_to_learn, pattern_to_learn_density, compute_loss, inject_pattern_step!


    const LOGIC_GATES = Dict(
    :FALSE => [0, 0, 0, 0],
    :AND   => [0, 0, 0, 1],
    :A_AND_NOT_B => [0, 0, 1, 0],
    :A     =>[0, 0, 1, 1],
    :NOT_A_AND_B =>[0, 1, 0, 0],
    :B     =>[0, 1, 0, 1],
    :XOR   => [0, 1, 1, 0],
    :OR    => [0, 1, 1, 1],
    :NOR   =>[1, 0, 0, 0],
    :XNOR  =>[1, 0, 0, 1],
    :NOT_B =>[1, 0, 1, 0],
    :A_OR_NOT_B => [1, 0, 1, 1],
    :NOT_A => [1, 1, 0, 0],
    :NOT_A_OR_B => [1, 1, 0, 1],
    :NAND  => [1, 1, 1, 0],
    :TRUE  =>[1, 1, 1, 1]
)



function pattern_to_learn(gate_name::Symbol; 
                          epochs = 5,          # Nombre de fois qu'on répète les 4 états pour apprendre
                          dt = 0.1, 
                          delay_io = 5.0,      # Délai physique de causalité
                          interval = 50.0,     # Temps de "respiration" entre deux tests
                          test_epochs = 2)     # Nombre de tests à la fin sans Output forcé
    
    # Récupération de la table de vérité
    truth_table = LOGIC_GATES[gate_name]
    base_states =[(0,0), (0,1), (1,0), (1,1)]
    
    # Calcul de la taille des tableaux
    total_lessons = (epochs + test_epochs) * 4
    total_time = total_lessons * interval + 50.0 # +50 de marge
    time_steps = 0:dt:total_time
    
    # Initialisation des 4 canaux (0.0 = repos, 1.0 = spike)
    Input_A  = zeros(length(time_steps))
    Input_B  = zeros(length(time_steps))
    Output_0 = zeros(length(time_steps))
    Output_1 = zeros(length(time_steps))
    
    current_time = 20.0 # On commence à t=20s
    
    # --- PHASE 1 : ENTRAÎNEMENT (Avec Output Forcé) ---
    for epoch in 1:epochs
        # On mélange pour éviter l'apprentissage de séquence temporelle pure
        shuffled_indices = shuffle(Random.default_rng(), 1:4)
        
        for idx in shuffled_indices
            A, B = base_states[idx]
            target = truth_table[idx]
            
            # Index temporel pour les Inputs
            idx_in = findfirst(>=(current_time), time_steps)
            Input_A[idx_in] = A
            Input_B[idx_in] = B
            
            # Index temporel pour la récompense/forçage (avec Délai)
            idx_out = findfirst(>=(current_time + delay_io), time_steps)
            
            if target == 0
                Output_0[idx_out] = 1.0 # On force le noeud "0"
            else
                Output_1[idx_out] = 1.0 # On force le noeud "1"
            end
            
            current_time += interval
        end
    end
    
    # --- PHASE 2 : TEST (Sans Output Forcé, c'est au réseau de deviner) ---
    test_target_mask =[] # Pour garder en mémoire les cibles pour ta Loss Function
    
    for epoch in 1:test_epochs
        shuffled_indices = shuffle(Random.default_rng(), 1:4)
        
        for idx in shuffled_indices
            A, B = base_states[idx]
            target = truth_table[idx]
            
            idx_in = findfirst(>=(current_time), time_steps)
            Input_A[idx_in] = A
            Input_B[idx_in] = B
            
            # ATTENTION : On n'écrit PAS dans Output_0 ni Output_1 ici.
            # On enregistre juste ce qu'on attend pour le CMA-ES
            push!(test_target_mask, (time_target = current_time + delay_io, expected = target))
            
            current_time += interval
        end
    end
    
    return time_steps, Input_A, Input_B, Output_0, Output_1, test_target_mask
end

function pattern_to_learn_density(gate_name::Symbol;
                          epochs = 5,          # Nombre de fois qu'on répète les 4 états pour apprendre
                          test_epochs = 3,     # Nombre de tests à la fin sans Output forcé
                          dt = 0.1,
                          interval = 6000.0,     # Temps total alloué pour un essai (doit être > delay + duration)
                          delay_io = 980.0,      # Temps avant que la réponse ne soit attendue/forcée
                          stimulus_duration = 5.0, # Durée de la vague d'entrée
                          force_duration = 5.0,    # Durée du forçage de la sortie (teacher forcing)
                          eval_duration = 1000.0,  # Durée de la fenêtre d'évaluation de la loss (phase test)
                          max_density = 60.0)       # Intensité du burst (cohérent avec test_hebbian.jl)

    # Récupération de la table de vérité (suppose que LOGIC_GATES est défini)
    truth_table = LOGIC_GATES[gate_name]
    base_states = [(0,0), (0,1), (1,0), (1,1)]

    # Calcul sécurisé du temps total
    total_lessons = (epochs + test_epochs) * 4
    start_time = 20.0
    total_time = start_time + (total_lessons * interval) + 50.0 
    time_steps = 0:dt:total_time

    # Initialisation des 4 canaux (Fonctions continues de densité)
    Input_A  = zeros(length(time_steps))
    Input_B  = zeros(length(time_steps))
    Output_0 = zeros(length(time_steps))
    Output_1 = zeros(length(time_steps))

    current_time = start_time

    # --- PHASE 1 : ENTRAÎNEMENT (Avec Output Forcé - Teacher Forcing) ---
    for epoch in 1:epochs
        shuffled_indices = shuffle(Random.default_rng(), 1:4)

        for idx in shuffled_indices
            A, B = base_states[idx]
            target = truth_table[idx]

            # 1. Fenêtre temporelle pour les INPUTS
            idx_in_start = findfirst(>=(current_time), time_steps)
            idx_in_end   = findfirst(>=(current_time + stimulus_duration), time_steps)
            
            # Application de la densité (créneau) au lieu d'un spike pur
            Input_A[idx_in_start:idx_in_end] .= A * max_density
            Input_B[idx_in_start:idx_in_end] .= B * max_density

            # 2. Fenêtre temporelle pour la RÉCOMPENSE / FORÇAGE
            idx_out_start = findfirst(>=(current_time + delay_io), time_steps)
            idx_out_end   = findfirst(>=(current_time + delay_io + force_duration), time_steps)

            if target == 0
                Output_0[idx_out_start:idx_out_end] .= max_density
            else
                Output_1[idx_out_start:idx_out_end] .= max_density
            end

            # On avance le temps pour la prochaine leçon
            current_time += interval
        end
    end

    # --- PHASE 2 : TEST (Sans Output Forcé, c'est au réseau de deviner) ---
    test_target_mask =[] # Historique pour la Loss Function

    for epoch in 1:test_epochs
        shuffled_indices = shuffle(Random.default_rng(), 1:4)

        for idx in shuffled_indices
            A, B = base_states[idx]
            target = truth_table[idx]

            idx_in_start = findfirst(>=(current_time), time_steps)
            idx_in_end   = findfirst(>=(current_time + stimulus_duration), time_steps)
            
            Input_A[idx_in_start:idx_in_end] .= A * max_density
            Input_B[idx_in_start:idx_in_end] .= B * max_density

            # ATTENTION : On n'écrit PAS dans Output_0 ni Output_1 ici.
            # On enregistre une FENÊTRE d'évaluation pour la Loss Function
            eval_start = current_time + delay_io
            eval_end   = current_time + delay_io + eval_duration
            
            push!(test_target_mask, (
                t_start = eval_start, 
                t_end = eval_end, 
                expected = target
            ))

            current_time += interval
        end
    end

    return time_steps, Input_A, Input_B, Output_0, Output_1, test_target_mask
end



# ------------------------------------------------------------------------------
# compute_loss  (v2 — multi-termes)
#
# 5 termes combinés pour guider CMA-ES vers des réseaux actifs, propagateurs,
# mécaniquement vivants et discriminants avant de viser la classification pure.
#
#  L_BCE           (W=1.0) : cross-entropy softmax sur Output_0 / Output_1
#  L_silence       (W=0.3) : exp(-mean_D_global / K_DM) — punit le silence global
#  L_propagation   (W=0.3) : exp(-mean_D_inter  / K_DM) — punit si les interneurones
#                             ne reçoivent pas de signal
#  L_mech_M        (W=0.2) : exp(-mean_M / K_DM)       — punit si M ne s'accumule pas
#  L_mech_Eint     (W=0.2) : exp(-mean_Eint / K_DM)    — punit si E_int reste nul
#  L_discrimination(W=0.3) : exp(-mean|r0-r1|)          — punit si outputs indiscernables
#
# Ancienne version (BCE + SILENCE_ALPHA*exp(-SILENCE_BETA*(r0+r1))) conservée ci-dessous :
# const SILENCE_ALPHA = 0.5
# const SILENCE_BETA  = 1.0
# function compute_loss(history, time_steps, test_target_mask, node_roles, d_idx, K_DM)
#   ... bce + SILENCE_ALPHA * exp(-SILENCE_BETA * (r0 + r1)) ...
# end
# ------------------------------------------------------------------------------

const SPIKE_THRESHOLD = 100.0   # seuil de D (mol) pour compter un "spike actif"

function compute_loss(history::Array{Float32, 3},
                      time_steps,
                      test_target_mask::Vector,
                      node_roles::Dict{Symbol, Int},
                      d_idx::Int)::Float64

    isempty(test_target_mask) && return 0.0

    id_O0           = node_roles[:Output_0]
    id_O1           = node_roles[:Output_1]
    n_steps_history = size(history, 1)

    total_bce = 0.0
    n_valid   = 0

    for trial in test_target_mask
        i_start = findfirst(>=(trial.t_start), time_steps)
        i_end   = findfirst(>=(trial.t_end),   time_steps)
        (isnothing(i_start) || isnothing(i_end)) && continue
        i_end = min(i_end, n_steps_history)
        i_start > i_end && continue

        # Fraction du temps où D dépasse le seuil → "taux de décharge" ∈ [0, 1]
        r0 = mean(view(history, i_start:i_end, id_O0, d_idx) .> Float32(SPIKE_THRESHOLD))
        r1 = mean(view(history, i_start:i_end, id_O1, d_idx) .> Float32(SPIKE_THRESHOLD))

        m     = max(r0, r1)
        denom = exp(r0 - m) + exp(r1 - m)
        p1    = clamp(exp(r1 - m) / denom, 1e-7, 1.0 - 1e-7)

        expected = Float64(trial.expected)
        bce      = -(expected * log(p1) + (1.0 - expected) * log(1.0 - p1))

        total_bce += bce
        n_valid   += 1

        @printf "  [trial] exp=%d | r0=%.3f r1=%.3f | p1=%.3f | bce=%.3f\n" Int(expected) r0 r1 p1 bce
    end

    n_valid == 0 && return Inf
    loss = total_bce / n_valid

    @printf "  [loss] BCE=%.4f\n" loss

    return loss
end


# ------------------------------------------------------------------------------
# inject_pattern_step!
#
# À appeler dans le bloc === A. INPUTS === de la boucle de simulation,
# en remplacement des bursts stochastiques.
#
# Les vecteurs Input_A, Input_B, Output_0, Output_1 sont issus de
# pattern_to_learn_density et précalculés avant la boucle.
# Pendant l'entraînement Output_0/Output_1 sont non-nuls → teacher forcing.
# Pendant le test ils valent 0 → le réseau est libre.
# ------------------------------------------------------------------------------

function inject_pattern_step!(net::BioNetwork,
                               step::Int,
                               Input_A::Vector{Float64},
                               Input_B::Vector{Float64},
                               Output_0::Vector{Float64},
                               Output_1::Vector{Float64},
                               node_roles::Dict{Symbol, Int})

    step > length(Input_A) && return

    Input_A[step]  > 0.0 && add_input!(net.nodes[node_roles[:Input_A]],  :D, Input_A[step])
    Input_B[step]  > 0.0 && add_input!(net.nodes[node_roles[:Input_B]],  :D, Input_B[step])
    Output_0[step] > 0.0 && add_input!(net.nodes[node_roles[:Output_0]], :D, Output_0[step])
    Output_1[step] > 0.0 && add_input!(net.nodes[node_roles[:Output_1]], :D, Output_1[step])
end

end
