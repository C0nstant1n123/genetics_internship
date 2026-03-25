# Import Packages 
include("../src/BioKan.jl")
using .BioKan
using Catalyst, JumpProcesses, Distributions, Plots
using Distributions
include("../lib/rate-distortion-example/src/RateDistortionBiology.jl")
using .RateDistortionBiology

dt = 1/10
total_steps = Int(500000.0 / dt)
R_cell = 0.5e-6
input_node_id = 1 

species_names = [:X, :Y, :Y_trans, :Z, :Prom, :Prom_act, :mRNA] 
circuit_input, p_defaults_vec_input = create_genetic_simple_circuit_input_integrated(:node_generic)

params_dict_input = Dict(parameters(circuit_input) .=> p_defaults_vec_input)

#set_param!(params_dict_input, :k_create_X, 0.001)
#set_param!(params_dict_input, :k_deg_X, 2.9e-5)

#  Build Bacteries
u0_dict_raw_source = Dict(:X => 30.0, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :Prom => 1.0, :Prom_act => 0.0, :mRNA => 0.0)



u0_source_dict = map_symbols_to_species(circuit_input, u0_dict_raw_source)


b1 = Bacterium(1, [0.003100, 0.005], circuit_input, params_dict_input, u0_source_dict)


history_all = zeros(total_steps)
for step in 1:total_steps
    t_sim = step * dt
    
    step_bacterium!(b1, dt) 
    history_all[step] = max(0.0, get_species(b1,:X))
    if step % 1000 == 0
        println(step)
    end

    end


times = (1:total_steps) .* dt

p1 = plot(times, history_all,   
    title  = "Bactérie 1 (Source)",
    label  = "X",
    color  = :blue,
    lw     = 2,
    xlabel = "Temps (s)",
    ylabel = "Molécules X"
)



