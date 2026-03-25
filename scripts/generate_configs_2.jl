using JLD2
using DrWatson
using Catalyst
using ModelingToolkit
using Random # Pour gérer les seeds

# On charge BioKan
# (Ajuste le chemin si besoin selon où tu lances le script)
include("../src/BioKan.jl") 
using .BioKan

# ==============================================================================
# PARAMÈTRES GLOBAUX
# ==============================================================================
N_REPLICAS = 3 # <--- C'EST ICI QUE TU CHOISIS LE NOMBRE DE RÉPÉTITIONS !

# ==============================================================================
# 1. CONFIGURATION DE BASE
# ==============================================================================
function get_base_config()
    ratio = 100
    return Dict(
        :experiment_name => "default_experiment",
        :sim => Dict(
            :dt => 0.1,
            :T_total => 1000.0,
            :save_every => 100
        ),
        :topology => Dict(
            :n_bacteries => 2,
            :distance_comm => 100, 
            :R_cell => 0.5e-6,
            :pos_input => [0.003100, 0.001],
            :pos_output => [0.003110, 0.001]
        ),
        :env => Dict(
            :D_defaults => Dict(:X => 0.0, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :mRNA => 0.0),
            :gamma_defaults => Dict(:X => 2.9e-3, :Y => 2.9e-3, :Y_trans => 2.9e-3, :Z => 2.9e-3, :mRNA => 0.0)
        ),
        :bio_params => Dict(
            :k_deg_X_ref => 9e-5,
            :ratio_input => ratio, 
            :u0_source => Dict(:X => ratio, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :mRNA => 0.0),
            :u0_receiv => Dict(:X => 0.0, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :mRNA => 0.0)
        )
    )
end

# ==============================================================================
# 2. LE SWEEP
# ==============================================================================
function get_sweep_list()
    sweep_dict = Dict(
        :v_max => 10 .^ range(log10(0.01), log10(1), length=2),
        :K_input => [100.0],
        :K_output => [1.0],
        :k_sec => [0.0],
        :k_transl => collect(range(0.05, 1, length=2))
    )
    return dict_list(sweep_dict) 
end

# ==============================================================================
# 3. UTILITAIRES (INCHANGÉS)
# ==============================================================================
function update_catalyst_param!(param_dict, param_name::Symbol, new_value)
    for (k, v) in param_dict
        if Symbol(k) == param_name
            param_dict[k] = new_value
        end
    end
end

function apply_sweep_logic!(config, params_list, sweep_key, sweep_val)
    if sweep_key == :ratio_input
        config[:bio_params][:ratio_input] = sweep_val
        k_deg = config[:bio_params][:k_deg_X_ref] 
        k_create = k_deg * sweep_val
        update_catalyst_param!(params_list[:source], :k_deg_X, k_deg)
        update_catalyst_param!(params_list[:source], :k_create_X, k_create)
        config[:bio_params][:u0_source][:X] = sweep_val
    else
        update_catalyst_param!(params_list[:source], sweep_key, sweep_val)
        update_catalyst_param!(params_list[:receiver], sweep_key, sweep_val)
        config[:bio_params][sweep_key] = sweep_val
    end
end

function create_empty_rn(type)
    # Helper pour recréer proprement les réseaux à chaque boucle
    if type == :source
        return BioKan.create_genetic_hill_repeter_input(:source)
    else
        return BioKan.create_genetic_hill_repeter_output(:receiver)
    end
end

# ==============================================================================
# 4. ASSEMBLAGE AUTOMATIQUE (AVEC RÉPLICATS)
# ==============================================================================
function build_final_configs()
    base = get_base_config()
    sweeps = get_sweep_list()
    final_configs = []

    println("⚙️  Génération : $(length(sweeps)) combinaisons × $N_REPLICAS réplicats...")
    
    global_counter = 1 # Pour avoir un ID unique par fichier (sim_0001, sim_0002...)

    # BOUCLE 1 : Les paramètres physiques
    for (group_idx, sweep_instance) in enumerate(sweeps)

        # BOUCLE 2 : Les réplicats statistiques
        for replica_idx in 1:N_REPLICAS
            
            # 1. Copie de base
            config = deepcopy(base)

            # 2. Création RN
            rn_input, p_vec_input = create_empty_rn(:source)
            rn_output, p_vec_output = create_empty_rn(:receiver)

            params_dicts = Dict(
                :source => Dict(parameters(rn_input) .=> p_vec_input),
                :receiver => Dict(parameters(rn_output) .=> p_vec_output)
            )

            # 3. Application logique métier
            apply_sweep_logic!(config, params_dicts, :ratio_input, config[:bio_params][:ratio_input])

            meta_data = Dict()
            for (key, val) in sweep_instance
                apply_sweep_logic!(config, params_dicts, key, val)
                meta_data[key] = val
            end

            # 4. Assemblage Final
            final_config = Dict(
                :id => global_counter,       # ID unique fichier
                :group_id => group_idx,      # ID du set de paramètres (commun aux 20 réplicats)
                :replica_id => replica_idx,  # Numéro du réplicat (1..20)
                :seed => rand(UInt64),       # GRAINE ALÉATOIRE UNIQUE 🎲
                
                :meta => meta_data,
                :sim => config[:sim],
                :topology => config[:topology],
                :env => config[:env],
                :biology => Dict(
                    :species_names => [:X, :Y, :Y_trans, :Z, :mRNA],
                    :circuits => Dict(:source => rn_input, :receiver => rn_output),
                    :parameters => Dict(:source => params_dicts[:source], :receiver => params_dicts[:receiver]),
                    :u0 => Dict(:source => config[:bio_params][:u0_source], :receiver => config[:bio_params][:u0_receiv])
                )
            )

            push!(final_configs, final_config)
            global_counter += 1
        end
    end

    return final_configs
end

# ==============================================================================
# MAIN
# ==============================================================================
configs = build_final_configs()

if !isdir("inputs") mkdir("inputs") end
output_path = "inputs/experiment_batch_01.jld2"
save_object(output_path, configs)

println("✅ Sauvegarde terminée : $output_path")
println("📊 Total simulations à lancer : $(length(configs))")
