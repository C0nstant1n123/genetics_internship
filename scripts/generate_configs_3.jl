using JLD2
using DrWatson
using Catalyst
using ModelingToolkit
using Random

# Chargement du module 
include("../src/BioKan.jl") 
using .BioKan

# ==============================================================================
# PARAMÈTRES GLOBAUX
# ==============================================================================
N_REPLICAS = 10           # Nombre de répétitions par config
BATCH_NAME = "batch_04"  # Nom de l'expérience globale

# ==============================================================================
# 1. CONFIGURATION DE BASE
# ==============================================================================
function get_base_config()
    ratio = 100
    return Dict(
        :experiment_name => "bio_kan_reservoir",
        :sim => Dict(
            :dt => 0.1,
            :T_total => 1000000.0,
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
# 2. LE SWEEP (Paramètres variables)
# ==============================================================================
function get_sweep_list()
    sweep_dict = Dict(
        :v_max => 0.1,
        :k_transl => collect(range(1, 10, length=16))
    )
    return dict_list(sweep_dict) 
end

# ==============================================================================
# 3. UTILITAIRES
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
    if type == :source
        return BioKan.create_genetic_hill_repeter_input(:source)
    else
        return BioKan.create_genetic_hill_repeter_output(:receiver)
    end
end

# ==============================================================================
# 4. ASSEMBLAGE AUTOMATIQUE
# ==============================================================================
function build_final_configs()
    base = get_base_config()
    sweeps = get_sweep_list()
    final_configs = []

    println("⚙️  Génération : $(length(sweeps)) combinaisons × $N_REPLICAS réplicats...")

    global_counter = 1 

    for (group_idx, sweep_instance) in enumerate(sweeps)

        # ---------------------------------------------------------
        # A. Construction du nom de dossier basé sur les paramètres
        # ---------------------------------------------------------
        folder_parts =String[]
        # On trie les clés pour que l'ordre soit toujours le même (a=1_b=2 et pas b=2_a=1)
        for key in sort(collect(keys(sweep_instance)))
            val = sweep_instance[key]
            # On formatte un peu pour éviter les floats à 15 décimales dans le nom de dossier
            val_str = isa(val, AbstractFloat) ? round(val, digits=4) : val
            push!(folder_parts, "$(key)=$(val_str)")
        end
        folder_name = join(folder_parts, "_")

        # ---------------------------------------------------------
        # B. Création des réplicats
        # ---------------------------------------------------------
        for replica_idx in 1:N_REPLICAS
            config = deepcopy(base)

            # Re-création propre des réseaux Catalyst
            rn_input, p_vec_input = create_empty_rn(:source)
            rn_output, p_vec_output = create_empty_rn(:receiver)

            params_dicts = Dict(
                :source => Dict(parameters(rn_input) .=> p_vec_input),
                :receiver => Dict(parameters(rn_output) .=> p_vec_output)
            )

            # Application des paramètres
            apply_sweep_logic!(config, params_dicts, :ratio_input, config[:bio_params][:ratio_input])

            meta_data = Dict()
            for (key, val) in sweep_instance
                apply_sweep_logic!(config, params_dicts, key, val)
                meta_data[key] = val
            end

            # Assemblage Final
            final_config = Dict(
                :id => global_counter,
                :group_id => group_idx,
                :replica_id => replica_idx,
                :seed => rand(UInt64),
                
                # C'est ici qu'on stocke le nom du dossier pour le Runner
                :folder_name => folder_name, 

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
output_path = "inputs/experiment_$(BATCH_NAME).jld2"
save_object(output_path, configs)

println("✅ Sauvegarde terminée : $output_path")
println("📂 Dossier exemple : outputs/$(BATCH_NAME)/$(configs[1][:folder_name])")
println("📊 Total simulations à lancer : $(length(configs))")

