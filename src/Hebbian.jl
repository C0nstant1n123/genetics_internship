module Hebbian

    using ..Bacterias
    using ..BioNetworks
    using Random
    using Statistics
    using Printf

    export pattern_to_learn, pattern_to_learn_density, compute_loss, compute_loss_spiking, inject_pattern_step!,
           pattern_to_learn_conditioning, compute_loss_conditioning, inject_pattern_step_conditioning!,
           pattern_to_learn_reversal, pattern_to_learn_xor, LOGIC_GATES


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
                          epochs = 10,          # Nombre de fois qu'on répète les 4 états pour apprendre
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
                          epochs = 12,          # Nombre de fois qu'on répète les 4 états pour apprendre
                          test_epochs = 3,     # Nombre de tests à la fin sans Output forcé
                          dt = 0.1,
                          interval = 8000.0,     # Temps total alloué pour un essai (doit être > delay + duration)
                          delay_io = 0.0,      # Temps avant que la réponse ne soit attendue/forcée
                          stimulus_duration = 10.0, # Durée de la vague d'entrée
                          force_duration = 10.0,    # Durée du forçage de la sortie (teacher forcing)
                          eval_duration = 3000.0,  # Durée de la fenêtre d'évaluation de la loss (phase test)
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
# compute_loss_spiking  (v3 — BCE + kurtosis + fréquence)
#
#  L_BCE    (w=1.0) : cross-entropy softmax sur Output_0 / Output_1 (même que v2)
#  L_kurt   (w=0.2) : exp(-excess_kurtosis / 5) — punit les signaux D plats (non spike)
#                     kurtosis excédentaire élevé → signal spike → L_kurt petit
#  L_freq   (w=0.3) : max(0, f_D - f_input) / f_input
#                     pénalise si la fréquence moyenne de spikes de D dépasse
#                     la fréquence des inputs — aucune pénalité si f_D ≤ f_input
# ------------------------------------------------------------------------------

function _excess_kurtosis(x::AbstractVector)::Float64
    n = length(x)
    n < 4 && return 0.0
    μ = mean(x)
    σ = std(x)
    σ < 1e-10 && return 0.0   # signal constant → kurtosis indéfini, on ignore
    return mean(((x .- μ) ./ σ).^4) - 3.0
end

function _count_spikes(trace::AbstractVector, threshold::Float64)::Int
    n = 0
    above = false
    for v in trace
        if !above && v > threshold
            n += 1; above = true
        elseif above && v <= threshold
            above = false
        end
    end
    return n
end

function compute_loss_spiking(history::Array{Float32, 3},
                               time_steps,
                               test_target_mask::Vector,
                               node_roles::Dict{Symbol, Int},
                               d_idx::Int,
                               Input_A::Vector{Float64},
                               Input_B::Vector{Float64};
                               w_bce      = 1.0,
                               w_kurtosis = 0.2,
                               w_freq     = 0.3)::Float64

    isempty(test_target_mask) && return 0.0

    n_steps_history = size(history, 1)
    n_nodes         = size(history, 2)
    id_O0           = node_roles[:Output_0]
    id_O1           = node_roles[:Output_1]

    # ── Terme 1 : BCE (identique à compute_loss) ────────────────────────────
    total_bce = 0.0
    n_valid   = 0
    for trial in test_target_mask
        i_start = findfirst(>=(trial.t_start), time_steps)
        i_end   = findfirst(>=(trial.t_end),   time_steps)
        (isnothing(i_start) || isnothing(i_end)) && continue
        i_end = min(i_end, n_steps_history)
        i_start > i_end && continue

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
    L_bce = total_bce / n_valid

    # ── Terme 2 : Kurtosis (sur toute la simulation, tous les nœuds) ─────────
    mean_kurt = mean(
        _excess_kurtosis(Float64.(view(history, :, i, d_idx)))
        for i in 1:n_nodes
    )
    L_kurt = exp(-max(0.0, mean_kurt) / 5.0)   # ∈ (0,1], → 0 si kurtosis élevé (bon)

    # ── Terme 3 : Fréquence spikes D vs fréquence inputs ────────────────────
    # Fréquence input : nombre d'événements (montées 0→>0) / durée totale
    T_total = Float64(time_steps[end] - time_steps[1])
    n_input_events = 0
    for i in 2:length(Input_A)
        (Input_A[i] > 0.0 && Input_A[i-1] == 0.0) && (n_input_events += 1)
        (Input_B[i] > 0.0 && Input_B[i-1] == 0.0) && (n_input_events += 1)
    end
    f_input = n_input_events / T_total

    # Fréquence moyenne spikes D sur tous les nœuds
    f_D = mean(_count_spikes(Float64.(view(history, :, i, d_idx)), SPIKE_THRESHOLD) / T_total
               for i in 1:n_nodes)

    L_freq = max(0.0, f_D - f_input) / (f_input + 1e-10)   # 0 si f_D ≤ f_input

    loss = w_bce * L_bce + w_kurtosis * L_kurt + w_freq * L_freq

    @printf "  [loss] BCE=%.4f | L_kurt=%.4f (kurt=%.2f) | L_freq=%.4f (f_D=%.4f f_in=%.4f)\n" L_bce L_kurt mean_kurt L_freq f_D f_input
    @printf "  [loss] TOTAL=%.4f\n" loss

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

# ------------------------------------------------------------------------------
# pattern_to_learn_conditioning
#
# Deux inputs : Input_A (CS+) et Input_B (CS-).
# Si Input_A → Output doit s'activer (expected=1)
# Si Input_B → Output doit rester silencieux (expected=0)
# Un seul nœud output.
# ------------------------------------------------------------------------------
function pattern_to_learn_conditioning(;
                          epochs          = 30,
                          test_epochs     = 20,
                          pause_epochs    = 2,      # pause silencieuse entre entraînement et test
                          dt              = 0.1,
                          interval        = 5000.0,
                          delay_io        = 2000.0,
                          stimulus_duration = 2000.0,
                          force_duration  = 2000.0,
                          eval_duration   = 5000.0,
                          max_density     = 50.0,
                          cs_plus         = :A)     # :A → Input_A est le CS+, :B → Input_B est le CS+

    cs_minus = (cs_plus == :A) ? :B : :A

    # 2 stimuli par epoch d'entraînement, 2 par epoch de test, espacés de 2×interval en test
    total_time = 1e4 +
                 epochs * 2 * interval +
                 pause_epochs * interval +
                 test_epochs * 2 * 2 * interval +   # chaque essai de test espacé de 2×interval
                 50.0
    start_time = 1e4
    time_steps = 0:dt:total_time

    Input_A  = zeros(length(time_steps))
    Input_B  = zeros(length(time_steps))
    Output   = zeros(length(time_steps))

    current_time = start_time

    # PHASE 1 : entraînement — alternance CS+ / CS- mélangée
    for _ in 1:epochs
        for cs in shuffle(Random.default_rng(), [cs_plus, cs_minus])
            i_in_start = findfirst(>=(current_time),                     time_steps)
            i_in_end   = findfirst(>=(current_time + stimulus_duration),  time_steps)
            inp = (cs == :A) ? Input_A : Input_B
            inp[i_in_start:i_in_end] .= max_density

            if cs == cs_plus
                i_out_start = findfirst(>=(current_time + delay_io),                  time_steps)
                i_out_end   = findfirst(>=(current_time + delay_io + force_duration), time_steps)
                Output[i_out_start:i_out_end] .= max_density
            end

            current_time += interval
        end
    end

    # PAUSE — silence entre entraînement et test
    current_time += pause_epochs * interval

    # PHASE 2 : test — plus de forcing, essais espacés de 2×interval
    test_target_mask = []
    freeze_intervals = NamedTuple{(:t_start, :t_end), Tuple{Float64, Float64}}[]
    t_freeze_start   = current_time
    for _ in 1:test_epochs
        for cs in shuffle(Random.default_rng(), [cs_plus, cs_minus])
            i_in_start = findfirst(>=(current_time),                     time_steps)
            i_in_end   = findfirst(>=(current_time + stimulus_duration),  time_steps)
            inp = (cs == :A) ? Input_A : Input_B
            inp[i_in_start:i_in_end] .= max_density

            push!(test_target_mask, (
                t_start  = current_time,
                t_end    = current_time + eval_duration,
                expected = (cs == cs_plus) ? 1 : 0
            ))

            current_time += 2 * interval   # 1 epoch de plus entre chaque essai de test
        end
    end
    push!(freeze_intervals, (t_start = t_freeze_start, t_end = current_time))

    return time_steps, Input_A, Input_B, Output, test_target_mask, freeze_intervals
end

# ------------------------------------------------------------------------------
# compute_loss_conditioning — BCE binaire sur un seul output
# ------------------------------------------------------------------------------
function compute_loss_conditioning(history::Array{Float32, 3},
                                   time_steps,
                                   test_target_mask::Vector,
                                   node_roles::Dict{Symbol, Int},
                                   d_idx::Int)::Float64

    isempty(test_target_mask) && return 0.0

    id_out          = node_roles[:Output]
    n_steps_history = size(history, 1)
    total_bce       = 0.0
    n_valid         = 0

    for trial in test_target_mask
        i_start = findfirst(>=(trial.t_start), time_steps)
        i_end   = findfirst(>=(trial.t_end),   time_steps)
        (isnothing(i_start) || isnothing(i_end)) && continue
        i_end = min(i_end, n_steps_history)
        i_start > i_end && continue

        # Fraction du temps où D > seuil → "taux de décharge" ∈ [0,1]
        p = Float64(mean(view(history, i_start:i_end, id_out, d_idx) .> Float32(SPIKE_THRESHOLD)))
        p = clamp(p, 1e-7, 1.0 - 1e-7)

        expected = Float64(trial.expected)
        bce      = -(expected * log(p) + (1.0 - expected) * log(1.0 - p))

        total_bce += bce
        n_valid   += 1

        @printf "  [trial] exp=%d | p=%.3f | bce=%.3f\n" Int(expected) p bce
    end

    n_valid == 0 && return Inf
    loss = total_bce / n_valid
    @printf "  [loss] BCE=%.4f\n" loss
    return loss
end

# ------------------------------------------------------------------------------
# inject_pattern_step_conditioning! — version pour un seul output
# ------------------------------------------------------------------------------
function inject_pattern_step_conditioning!(net::BioNetwork,
                                           step::Int,
                                           Input_A::Vector{Float64},
                                           Input_B::Vector{Float64},
                                           Output::Vector{Float64},
                                           node_roles::Dict{Symbol, Int})
    step > length(Input_A) && return
    set_species!(net.nodes[node_roles[:Input_A]], :D, Input_A[step])
    set_species!(net.nodes[node_roles[:Input_B]], :D, Input_B[step])
    set_species!(net.nodes[node_roles[:Output]],  :D, Output[step])
end

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

# ------------------------------------------------------------------------------
# pattern_to_learn_reversal
#
# Protocole de conditionnement inversé en deux phases :
#   Phase 1 (epochs trials) : CS+_A (Input_A + Output forcé), CS-_B (Input_B seul)
#   Test 1  (test_epochs)   : A et B sans forcing — S gelé
#   Phase 2 (epochs trials) : CS+_B (Input_B + Output forcé), CS-_A (Input_A seul)
#   Test 2  (test_epochs)   : A et B sans forcing — S gelé
#
# Retourne :
#   time_steps, Input_A, Input_B, Output, mask, freeze_intervals
#
# freeze_intervals : Vector de NamedTuples (t_start, t_end) couvrant chaque phase de test.
# La boucle de simulation doit geler S dans ces intervalles.
# ------------------------------------------------------------------------------
function pattern_to_learn_reversal(;
                          epochs            = 70,
                          test_epochs       = 10,
                          pause_epochs      = 2,
                          dt                = 1.0,
                          interval          = 5000.0,
                          delay_io          = 1000.0,
                          eval_delay        = 0.0,
                          stimulus_duration = 2000.0,
                          force_duration    = 2000.0,
                          eval_duration     = 5000.0,
                          max_density       = 50.0)

    total_time = 1e4 +
                 epochs       * 2 * interval +
                 pause_epochs * interval +
                 test_epochs  * 2 * 2 * interval +
                 pause_epochs * interval +
                 epochs       * 2 * interval +
                 pause_epochs * interval +
                 test_epochs  * 2 * 2 * interval +
                 50.0

    time_steps = 0:dt:total_time
    n          = length(time_steps)
    Input_A    = zeros(n)
    Input_B    = zeros(n)
    Output     = zeros(n)
    mask       = []
    freeze_intervals = NamedTuple{(:t_start, :t_end), Tuple{Float64, Float64}}[]

    current_time = Ref(1e4)   # Ref pour mutation dans les closures

    function fill_window!(vec, t_from, dur)
        i_s = findfirst(>=(t_from),       time_steps)
        i_e = findfirst(>=(t_from + dur), time_steps)
        (isnothing(i_s) || isnothing(i_e)) && return
        vec[i_s:i_e] .= max_density
    end

    function train_phase!(cs_plus::Symbol)
        cs_minus = (cs_plus == :A) ? :B : :A
        inp_plus  = (cs_plus  == :A) ? Input_A : Input_B
        inp_minus = (cs_minus == :A) ? Input_A : Input_B
        for _ in 1:epochs
            for cs in shuffle(Random.default_rng(), [cs_plus, cs_minus])
                if cs == cs_plus
                    fill_window!(inp_plus,  current_time[], stimulus_duration)
                    fill_window!(Output,    current_time[] + delay_io, force_duration)
                else
                    fill_window!(inp_minus, current_time[], stimulus_duration)
                end
                current_time[] += interval
            end
        end
    end

    function test_phase!(cs_plus::Symbol)
        t_freeze_start = current_time[]
        for _ in 1:test_epochs
            for cs in shuffle(Random.default_rng(), [:A, :B])
                expected = (cs == cs_plus) ? 1 : 0
                fill_window!(cs == :A ? Input_A : Input_B, current_time[], stimulus_duration)
                push!(mask, (
                    t_start  = current_time[] + eval_delay,
                    t_end    = current_time[] + eval_delay + eval_duration,
                    expected = expected
                ))
                current_time[] += 2 * interval
            end
        end
        push!(freeze_intervals, (t_start = t_freeze_start, t_end = current_time[]))
    end

    train_phase!(:A)
    current_time[] += pause_epochs * interval
    test_phase!(:A)
    current_time[] += pause_epochs * interval
    train_phase!(:B)
    current_time[] += pause_epochs * interval
    test_phase!(:B)

    return time_steps, Input_A, Input_B, Output, mask, freeze_intervals
end

# ------------------------------------------------------------------------------
# pattern_to_learn_xor
#
# Protocole XOR : 4 combinaisons d'entrées, 1 phase d'entraînement + 1 phase de test.
#   A seul  (expected=1) → Output forcé
#   B seul  (expected=1) → Output forcé
#   A+B     (expected=0) → pas de forcing (inhibition attendue)
#   aucun   (expected=0) → pas de forcing
#
# Retourne :
#   time_steps, Input_A, Input_B, Output, mask, freeze_intervals
# ------------------------------------------------------------------------------
"""
    pattern_to_learn_xor(; logical_gate, ...)

Génère un protocole d'entraînement/test pour n'importe quelle porte logique à 2 entrées.

`logical_gate` est un Dict définissant la sortie attendue pour chaque combinaison :
    logical_gate = Dict(:A_only=>1, :B_only=>1, :AB=>0, :none=>0)  # XOR (défaut)
    logical_gate = Dict(:A_only=>1, :B_only=>1, :AB=>1, :none=>0)  # OR
    logical_gate = Dict(:A_only=>0, :B_only=>0, :AB=>1, :none=>0)  # AND
    logical_gate = Dict(:A_only=>1, :B_only=>1, :AB=>0, :none=>1)  # XNOR
"""
function pattern_to_learn_xor(;
                      logical_gate      = :XOR,
                      epochs            = 0,
                      test_epochs       = 10,
                      pause_epochs      = 1,
                      dt                = 1.0,
                      interval          = 5000.0,
                      delay_io          = 1000.0,
                      stimulus_duration = 2000.0,
                      force_duration    = 2000.0,
                      eval_duration     = 5000.0,
                      max_density       = 50.0)

    # Convertir Symbol → Dict si nécessaire (ordre : none, B_only, A_only, AB)
    if logical_gate isa Symbol
        v = LOGIC_GATES[logical_gate]
        logical_gate = Dict(:none=>v[1], :B_only=>v[2], :A_only=>v[3], :AB=>v[4])
    end

    total_time = 1e4 +
                 epochs       * 4 * interval +
                 pause_epochs * interval +
                 test_epochs  * 4 * 2 * interval +
                 50.0

    time_steps = 0:dt:total_time
    n          = length(time_steps)
    Input_A    = zeros(n)
    Input_B    = zeros(n)
    Input_0    = zeros(n)
    Output     = zeros(n)
    mask       = []
    freeze_intervals = NamedTuple{(:t_start, :t_end), Tuple{Float64, Float64}}[]

    current_time = Ref(1e4)

    function fill_window!(vec, t_from, dur)
        i_s = findfirst(>=(t_from),       time_steps)
        i_e = findfirst(>=(t_from + dur), time_steps)
        (isnothing(i_s) || isnothing(i_e)) && return
        vec[i_s:i_e] .= max_density
    end

    combos = [:A_only, :B_only, :AB, :none]
    O_silent_A      = zeros(n)   # spike O de B1 quand A est absent pendant un stimulus
    O_silent_B      = zeros(n)   # spike O de B3 quand B est absent pendant un stimulus
    O_silent_output = zeros(n)   # spike O de B6 quand output attendu = 0 (dépend de la porte)
    train_mask = NamedTuple{(:t_start, :t_end, :combo, :expected), Tuple{Float64,Float64,Symbol,Int}}[]

    # Phase d'entraînement
    for _ in 1:epochs
        for combo in shuffle(Random.default_rng(), combos)
            (combo == :A_only || combo == :AB) && fill_window!(Input_A,   current_time[], stimulus_duration)
            (combo == :B_only || combo == :AB) && fill_window!(Input_B,   current_time[], stimulus_duration)
            combo == :none                     && fill_window!(Input_0,   current_time[], stimulus_duration)
            logical_gate[combo] == 1           && fill_window!(Output,    current_time[] + delay_io, force_duration)
            # Input absent → spike O (signal d'absence, même durée que le stimulus)
            (combo == :B_only || combo == :none) && fill_window!(O_silent_A, current_time[], stimulus_duration)
            (combo == :A_only || combo == :none) && fill_window!(O_silent_B, current_time[], stimulus_duration)
            # Output attendu = 0 → B6 émet O (signal d'absence, dépend de la porte logique)
            logical_gate[combo] == 0           && fill_window!(O_silent_output, current_time[], stimulus_duration)
            push!(train_mask, (t_start=current_time[], t_end=current_time[]+interval, combo=combo, expected=logical_gate[combo]))
            current_time[] += interval
        end
    end

    current_time[] += pause_epochs * interval

    # Phase de test — S gelé
    t_freeze_start = current_time[]
    for _ in 1:test_epochs
        for combo in shuffle(Random.default_rng(), combos)
            (combo == :A_only || combo == :AB) && fill_window!(Input_A,   current_time[], stimulus_duration)
            (combo == :B_only || combo == :AB) && fill_window!(Input_B,   current_time[], stimulus_duration)
            combo == :none                     && fill_window!(Input_0,   current_time[], stimulus_duration)
            (combo == :B_only || combo == :none) && fill_window!(O_silent_A, current_time[], stimulus_duration)
            (combo == :A_only || combo == :none) && fill_window!(O_silent_B, current_time[], stimulus_duration)
            logical_gate[combo] == 0           && fill_window!(O_silent_output, current_time[], stimulus_duration)
            push!(mask, (
                t_start  = current_time[],
                t_end    = current_time[] + eval_duration,
                expected = logical_gate[combo],
                combo    = combo
            ))
            current_time[] += 2 * interval
        end
    end
    push!(freeze_intervals, (t_start = t_freeze_start, t_end = current_time[]))

    return time_steps, Input_A, Input_B, Input_0, Output, O_silent_A, O_silent_B, O_silent_output, mask, train_mask, freeze_intervals
end

end
