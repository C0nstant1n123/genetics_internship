using Statistics
include("../lib/rate-distortion-example/src/RateDistortionBiology.jl")

using .RateDistortionBiology


# ==============================================================================
# HELPER : Extraction de signal (Basé sur ta fonction plot_sim)
# ==============================================================================
function extract_signal_from_data(data, bacteria_idx, species_id, t_window=nothing)
    # 1. Extraction brute
    history = data[:history] # [Temps, Bactéries, Espèces]
    t_vec   = data[:time]

    # 2. Filtrage Temporel (Identique à plot_sim)
    if !isnothing(t_window)
        t_start, t_end = t_window
        indices = findall(x -> t_start <= x <= t_end, t_vec)
        if isempty(indices)
            error("❌ Aucun point trouvé dans l'intervalle $t_window.")
        end
        # On découpe
        history = history[indices, :, :]
    end

    # 3. Résolution du Nom de l'espèce -> Index
    # (Copie de la logique robuste de plot_sim)
    config = data[:config]
    if haskey(config, :biology) && haskey(config[:biology], :species_names)
        raw_names = config[:biology][:species_names]
    else
        # Fallback si pas de noms définis
        raw_names = ["Esp $i" for i in 1:size(history, 3)]
    end
    all_names = string.(raw_names)

    # Trouve l'index
    if species_id isa String
        s_idx = findfirst(==(species_id), all_names)
    elseif species_id isa Symbol
        s_idx = findfirst(==(string(species_id)), all_names)
    else
        s_idx = species_id
    end

    if isnothing(s_idx)
        error("❌ Espèce '$species_id' introuvable.")
    end

    # 4. Retourne le vecteur temporel (1D)
    return history[:, bacteria_idx, s_idx]
end

# ==============================================================================
# FONCTION 1 : Calcul de la Distorsion Standardisée D
# ==============================================================================
"""
    compute_distortion(data, signal1_loc, signal2_loc; t_window=nothing)

Calcule la distorsion de forme (MSE sur Z-scores) entre deux signaux.
Args:
- `signal1_loc` : Tuple (index_bactérie, nom_ou_index_espèce) ex: (1, "Glc")
- `signal2_loc` : Tuple (index_bactérie, nom_ou_index_espèce) ex: (1, "GFP")
"""
function compute_distortion(data, signal1_loc::Tuple, signal2_loc::Tuple; t_window=nothing)
    
    # 1. Extraction
    s1 = extract_signal_from_data(data, signal1_loc[1], signal1_loc[2], t_window)
    s2 = extract_signal_from_data(data, signal2_loc[1], signal2_loc[2], t_window)

    # 2. Standardisation (Z-Score)
    # On centre (mu=0) et on réduit (sigma=1) pour comparer les FORMES
    # et ignorer les échelles d'unités (mM vs uM).
    z1 = (s1 .- mean(s1)) ./ std(s1)
    z2 = (s2 .- mean(s2)) ./ std(s2)

    # 3. Calcul de D (Erreur Quadratique Moyenne Standardisée)
    # D = E[(Zx - Zy)^2]
    # Note : Mathématiquement équivalent à 2*(1 - corrélation)
    D = mean((z1 .- z2).^2)
    
    return D
end

# ==============================================================================
# FONCTION 2 : Objectif Bio-Logique (J = MI - alpha * C)
# ==============================================================================
"""
    compute_bio_objective(data, input_loc, output_loc, alpha; t_window=nothing)

Calcule la fonction de coût lagrangienne : Information - Coût Métabolique.
Args:
- `input_loc` : Signal d'entrée (Environnement).
- `output_loc`: Signal de sortie (Réponse Bactérie).
- `alpha`     : Poids du coût métabolique (bits / nM).
"""
function compute_bio_objective(data, input_loc::Tuple, output_loc::Tuple, alpha::Float64; t_window=nothing)
    
    # 1. Extraction
    X = extract_signal_from_data(data, input_loc[1], input_loc[2], t_window)
    Y = extract_signal_from_data(data, output_loc[1], output_loc[2], t_window)

    # 2. Calcul de l'Information Mutuelle (I ou R)
    # Utilise ta librairie RateDistortionBiology
    # On suppose pas de lag ici (synchronisation instantanée ou intégrée)
    mi_bits = mutual_information_continuous(X, Y)

    # 3. Calcul du Coût Métabolique (C)
    # C'est la moyenne de l'OUTPUT (Y). C'est la bactérie qui paie pour produire Y.
    # On suppose que X est gratuit (fourni par l'environnement).
    cost_metabolic = mean(Y)

    # 4. Calcul de l'Objectif (J)
    # On veut MAXIMISER J (Max info, Min coût)
    objective = mi_bits - (alpha * cost_metabolic)

    return (objective=objective, info_bits=mi_bits, cost=cost_metabolic)
end
