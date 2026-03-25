using JLD2
using DrWatson
using Catalyst
using ModelingToolkit

# On charge BioKan
include("../src/BioKan.jl")
using .BioKan

# ==============================================================================
# 1. LA CONFIGURATION PAR DÉFAUT (INCHANGÉE)
# ==============================================================================
function get_base_config()
    ratio = 100
    return Dict(
        :experiment_name => "default_experiment",
        :sim => Dict(
            :dt => 0.1,
            :T_total => 100000.0,
            :save_every => 100
        ),
        :topology => Dict(
            :n_bacteries => 2,
            :distance_comm => 100, # Distance en microns
            :R_cell => 0.5e-6,
            :pos_input => [0.003100, 0.001],
            :pos_output => [0.003110, 0.001]
        ),
        :env => Dict(
            :D_defaults => Dict(:X => 0.0, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :mRNA => 0.0),   #2.4e-8
            :gamma_defaults => Dict(:X => 2.9e-3, :Y => 2.9e-3, :Y_trans => 2.9e-3, :Z => 2.9e-3, :mRNA => 0.0)
        ),
        :bio_params => Dict(
            :k_deg_X_ref => 9e-5,
            :ratio_input => ratio, # Valeur par défaut
            
            # Conditions Initiales
            :u0_source => Dict(:X => ratio, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :mRNA => 0.0),
            :u0_receiv => Dict(:X => 0.0, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :mRNA => 0.0)
        )
    )
end

# ==============================================================================
# 2. LE SWEEP (C'EST ICI QUE TU JOUES !)
# ==============================================================================
function get_sweep_list()
    # Tu peux mettre N'IMPORTE QUOI ici :
    # - Des noms de paramètres Catalyst (:K, :v_max, :n, :k_transl...)
    # - Ou la variable spéciale :ratio_input
    
    sweep_dict = Dict(

        
        # On fait varier K (sensibilité)
        :v_max => 10 .^ range(log10(0.01), log10(1), length=16),
        :K_input => [100.0],
        :K_output => [1.0],
        :k_sec => [0.0],
        :k_transl => collect(range(0.05,1,length = 16))
    )

    # DrWatson génère le produit cartésien (toutes les combinaisons)
    return dict_list(sweep_dict) 
end

# ==============================================================================
# 3. UTILITAIRES D'INJECTION (MOTEUR INTERNE)
# ==============================================================================

"""
Met à jour un paramètre dans un dictionnaire Catalyst de manière robuste.
"""
function update_catalyst_param!(param_dict, param_name::Symbol, new_value)
    found = false
    # Les clés Catalyst sont des objets symboliques complexes, on compare leur nom
    for (k, v) in param_dict
        if Symbol(k) == param_name
            param_dict[k] = new_value
            found = true
            # Pas de break, car un paramètre peut exister en plusieurs exemplaires 
            # (ex: K utilisé dans plusieurs réactions)
        end
    end
    # Optionnel : décommenter pour debug
    # if !found println("⚠️ Paramètre :$param_name introuvable.") end
end

"""
Logique métier : traduit les valeurs du sweep en paramètres physiques.
C'est ici qu'on gère les cas particuliers comme le Ratio.
"""
function apply_sweep_logic!(config, params_list, sweep_key, sweep_val)

    # --- CAS SPÉCIAL 1 : LE RATIO D'INPUT ---
    if sweep_key == :ratio_input
        # 1. Mise à jour de la config (le "plan")
        config[:bio_params][:ratio_input] = sweep_val
        
        # 2. Récupération de la référence de dégradation
        # C'est ici qu'on prend la valeur que tu as définie dans get_base_config()
        k_deg = config[:bio_params][:k_deg_X_ref] 

        # 3. Calcul physique de la création
        k_create = k_deg * sweep_val
        
        # --- MISE A JOUR DU MOTEUR PHYSIQUE (Catalyst) ---
        
        # A. On force la mise à jour de k_deg_X dans le système
        # (Sinon il garde la valeur par défaut du fichier BioKan.jl)
        update_catalyst_param!(params_list[:source], :k_deg_X, k_deg)
        
        # B. On met à jour k_create_X
        update_catalyst_param!(params_list[:source], :k_create_X, k_create)

        # C. Mise à jour des conditions initiales
        config[:bio_params][:u0_source][:X] = sweep_val
        
        # Debug optionnel
        # println("  -> Ratio appliqué : k_deg=$k_deg, k_create=$k_create")

    # --- CAS GÉNÉRAL (K, v_max, k_transl...) ---
    else
        # On l'applique partout (Source ET Receiver)
        update_catalyst_param!(params_list[:source], sweep_key, sweep_val)

        
        # Si tu veux que le Receiver soit identique à la Source, décommente :
        update_catalyst_param!(params_list[:receiver], sweep_key, sweep_val)

        # On le stocke dans bio_params pour trace
        config[:bio_params][sweep_key] = sweep_val
    end
end

# ==============================================================================
# 4. ASSEMBLAGE AUTOMATIQUE (NE PLUS TOUCHER)
# ==============================================================================
function build_final_configs()
    base = get_base_config()
    sweeps = get_sweep_list()

    final_configs = []

    println("⚙️  Assemblage de $(length(sweeps)) configurations...")

    for (i, sweep_instance) in enumerate(sweeps)

        # 1. Copie propre de la base
        config = deepcopy(base)

        # 2. Création des objets Catalyst (Vierges)
        rn_input, p_vec_input = create_genetic_hill_repeter_input(:source)
        rn_output, p_vec_output = create_genetic_hill_repeter_output(:receiver)

        
        
        # Conversion en Dictionnaires modifiables
        params_dicts = Dict(
            :source => Dict(parameters(rn_input) .=> p_vec_input),
            :receiver => Dict(parameters(rn_output) .=> p_vec_output)
        )
        
        # 3. Application par défaut du ratio de base (au cas où il n'est pas dans le sweep)
        # Cela assure que k_create est toujours calculé correctement
        apply_sweep_logic!(config, params_dicts, :ratio_input, config[:bio_params][:ratio_input])

        # 4. BOUCLE MAGIQUE : Application du Sweep
        # On itère sur chaque paire du dictionnaire sweep (ex: :K => 10.0)
        meta_data = Dict()
        for (key, val) in sweep_instance
            apply_sweep_logic!(config, params_dicts, key, val)
            meta_data[key] = val # On garde une trace pour les plots
        end

        # 5. Assemblage Final
        final_config = Dict(
            :id => i,
            :experiment_name => "batch_sweep_generic",
            :meta => meta_data, # Contient exactement ce qui a varié

            :sim => config[:sim],
            :topology => config[:topology],
            :env => config[:env],

            :biology => Dict(
                :species_names => [:X, :Y, :Y_trans, :Z, :mRNA],
                :circuits => Dict(
                    :source => rn_input,
                    :receiver => rn_output
                ),
                :parameters => Dict(
                    :source => params_dicts[:source],
                    :receiver => params_dicts[:receiver]
                ),
                :u0 => Dict(
                    :source => config[:bio_params][:u0_source],
                    :receiver => config[:bio_params][:u0_receiv]
                )
            )
        )

        push!(final_configs, final_config)
    end

    return final_configs
end

# ==============================================================================
# EXÉCUTION
# ==============================================================================

configs = build_final_configs()

if !isdir("inputs") mkdir("inputs") end
output_path = "inputs/experiment_batch_01.jld2"
save_object(output_path, configs)

println("✅ Sauvegarde terminée : $output_path")
println("   Exemple config 1 Meta : $(configs[2][:meta])")
println(" Nombre de configs :", length(configs))
