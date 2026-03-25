module BioKan 
    # 1. Inclure les fichiers 
    include("Bacterias.jl")
    include("Network.jl")
    include("Diffusion.jl")

    # 2. Importer pour ré-exporter
    using .Bacterias
    using .BioNetworks
    using .Diffusion

    # 3. Exportations globales pour l'utilisateur
    export Bacterium, create_genetic_simple_circuit_input, create_genetic_simple_circuit_output, create_iFFL_circuit, create_propagating_iFFL_circuit, step_bacterium!, get_species, add_input!, map_symbols_to_species, set_species!, create_genetic_simple_circuit_input_integrated, set_param!, create_simple_death_birth_model, create_genetic_hill_repeter_input, create_genetic_hill_repeter_output, create_burst_circuit, create_hebbian_model
    export BioNetwork, add_bacterium!, build_edges!
    export compute_diffusion_kernels_physics, update_diffusion!, compute_static_coupling_physics, propagate_signals_instantaneous!
end


