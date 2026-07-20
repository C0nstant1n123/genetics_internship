module BioKan
    # 1. Inclure les fichiers
    include("Bacterias.jl")
    include("Network.jl")
    include("Diffusion.jl")
    include("Hebbian.jl")

    # 2. Importer pour ré-exporter
    using .Bacterias
    using .BioNetworks
    using .Diffusion
    using .Hebbian

    # 3. Exportations globales pour l'utilisateur
    export Bacterium, create_genetic_simple_circuit_input, create_genetic_simple_circuit_output, create_iFFL_circuit, create_propagating_iFFL_circuit, step_bacterium!, get_species, add_input!, map_symbols_to_species, set_species!, create_genetic_simple_circuit_input_integrated, set_param!, create_simple_death_birth_model, create_genetic_hill_repeter_input, create_genetic_hill_repeter_output, create_burst_circuit, create_hebbian_model, create_hebbian_stable_model, create_hebbian_non_spike_model, make_spike_schedule, spike!, SpikeSchedule, notify_bacterium!
    export BioNetwork, add_bacterium!, build_edges!, plot_bionetwork, plot_bionetwork_3d, build_network_square!, build_network_cube!, assign_tetrahedral_roles, assign_conditioning_roles
    export compute_diffusion_kernels_physics, update_diffusion!, compute_static_coupling_physics, propagate_signals_instantaneous!, compute_fpt_kernels, propagate_signals_delayed!
    export pattern_to_learn, pattern_to_learn_density, inject_pattern_step!, compute_loss, compute_loss_spiking, pattern_to_learn_conditioning, compute_loss_conditioning, inject_pattern_step_conditioning!, pattern_to_learn_reversal, pattern_to_learn_xor, LOGIC_GATES
end


