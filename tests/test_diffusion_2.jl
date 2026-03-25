# Import Packages 
include("../src/BioKan.jl")
using .BioKan
using Catalyst, JumpProcesses, Distributions, Plots
using Distributions


# 0. Input Functions _______________________________________

function periodic_step_input_signal(t, high_val=100.0, low_val=0.0; period=1000.0, duty_cycle=0.5)
    cycle_pos = t % period
    
    if cycle_pos < (period * duty_cycle)
        return high_val
    else
        return low_val
    end
end

function next_poisson_signal(current_val, high_val, low_val, rate, dt)
    p_switch = rate * dt
    
    if rand() < p_switch
        if isapprox(current_val, high_val, atol=1e-5)
            return low_val
        else
            return high_val
        end
    else
        return current_val
    end

end

# 1. General parmeters _________________________________________________________

n_bacteries = 2
taille_espace = 0.1
distance_comm = 0.1   
dt = 1/60 
total_steps = Int(100.0 / dt)
R_cell = 0.5e-6
input_node_id = 1 




# 2. Build Network ____________________________________________________________

net = BioNetwork(distance_comm, 5) 

# Build circuits
species_names = [:X, :Y, :Y_trans, :Z, :Prom, :Prom_act, :mRNA] 
n_species = length(species_names) 

circuit_input, p_defaults_vec_input = create_genetic_simple_circuit_input_integrated(:node_generic)   # could be create_genetic_simple_circuit_input_integrated
circuit_output, p_defaults_vec_output = create_genetic_simple_circuit_output(:node_generic)

# Add Circuits Parrameters
params_dict_input = Dict(parameters(circuit_input) .=> p_defaults_vec_input)
params_dict_output = Dict(parameters(circuit_output) .=> p_defaults_vec_output)


#  Build Bacteries
u0_dict_raw_source = Dict(:X => 30.0, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :Prom => 1.0, :Prom_act => 0.0, :mRNA => 0.0)
u0_dict_raw_receiv = Dict(:X => 0.0, :Y => 0.0, :Y_trans => 0.0, :Z => 0.0, :Prom => 1.0, :Prom_act => 0.0, :mRNA => 0.0)


u0_source_dict = map_symbols_to_species(circuit_input, u0_dict_raw_source)
u0_receiv_dict = map_symbols_to_species(circuit_output, u0_dict_raw_receiv)

b1 = Bacterium(1, [0.003100, 0.005], circuit_input, params_dict_input, u0_source_dict)
b2 = Bacterium(2, [0.003200, 0.005], circuit_output, params_dict_output, u0_receiv_dict)

add_bacterium!(net, b1)
add_bacterium!(net, b2)

# Build edges
build_edges!(net)
println("Réseau : $(length(net.nodes)) bactéries.")
println("Distance B1->B2 : $(net.edges[2])")



# 3. Diffusion Kernels __________________________________________________________



# D in m²/min : Diffuion Parrameters
D_dict = Dict(
    :X => 0.0,     
    :Y => 0.0,
    :Y_trans => 2.4e-10,
    :Z => 0.0,  
    :Prom => 0.0,  
    :Prom_act => 0.0, 
    :mRNA => 0.0   
)

# Gamma in 1/min : Degradation Parrameters
gamma_dict = Dict(
    :X => 2.9e-3,    
    :Y => 2.9e-3, 
    :Y_trans => 0.0,   
    :Z => 2.9e-3,    
    :Prom => 0.0,  
    :Prom_act => 0.0, 
    :mRNA => 0.0

)

# Build kernel : Computes K(r, t) for each edge and each species, returns a Dict with keys (id_s, id_t) and values K_edge (matrix n_species x n_steps)

T_max = 0.01
dt_kernel = 0.01
T_max_kernel = T_max / dt

kernels, t_vec = compute_diffusion_kernels_physics(
    net.edges, D_dict, gamma_dict, species_names, dt, T_max_kernel, R_cell
)

#Debug 
k =kernels[(1, 2)][1, :]
println("Maximum du kernel pour X de B1 à B2 : ", maximum(kernels[(1, 2)][1, :]), "temps :", t_vec[argmax(kernels[(1, 2)][1, :])])
println("Maximum du kernel pour Y de B1 à B2 : ", maximum(kernels[(1, 2)][3, :]), "temps :", t_vec[argmax(kernels[(1, 2)][2, :])])
                               
println("Maximum du kernel pour Y de B1 à B1 : ", maximum(kernels[(1, 1)][2, :]), "temps :", t_vec[argmax(kernels[(1, 1)][2, :])])
# Extraction of retention factors for the first time step (t=0) for each species, which will be used to calculate how much of the internal stock is retained vs leaked at each step.

retention_factors = kernels[(1, 1)][:, 1]




# 4. Simulation Loop __________________________________________________________

# Initialisation

# Buffer for convolution

buffer = zeros(Float64, n_bacteries, n_species, length(t_vec))
current_ptr = 1

# Entry signal at t=0
signal_X = 0.0

# Historic for plotting
history_all = zeros(total_steps, n_bacteries, n_species)
history_production = zeros(total_steps, n_bacteries, n_species)



println("Simulation : Démarrage...")


for step in 1:total_steps
    global signal_X 
    t_sim = step * dt
    


    "
    # --- 1. Input Control (only for explicit inputs) ---
    if haskey(net.nodes, input_node_id)
        b_in = net.nodes[input_node_id]

        # Calcul Poisson
        signal_X = next_poisson_signal(signal_X, 100.0, 0.0, 0.1, dt)

        set_species!(b_in, :X, Float64(signal_X)) 
    end
    "
    

    # --- 2. Intern biology (Step for every bacteria)
    
    for (id, b) in net.nodes
        step_bacterium!(b, dt)
        
    end

    # --- 3. Diffusion outside 
    
    flux_emissions = zeros(n_bacteries, n_species)
    retained_stock = zeros(n_bacteries, n_species)

    for i in 1:n_bacteries
        if haskey(net.nodes, i)
            b = net.nodes[i] 
            
            for s in 1:n_species
                qty = max(0.0, get_species(b, species_names[s]))


                #retained = qty * retention_factors[s]
                #leaked   = qty * (1.0 - retention_factors[s])

                retained = max(0.0, rand(Binomial(round(Int, qty), retention_factors[s])))
                leaked = qty - retained

                retained_stock[i, s] = retained
                flux_emissions[i, s] = leaked
            end
        end
    end


    # --- 4. Diffusion inside
    received, next_ptr = update_diffusion!(buffer, kernels, flux_emissions, current_ptr)
    current_ptr = next_ptr

    # --- 5. Update internal states based on received signals and retained stock

    for i in 1:n_bacteries
        if haskey(net.nodes, i)
            b = net.nodes[i] 
            
            for s in 1:n_species
                sym = species_names[s]

                
                total_new = retained_stock[i, s] + received[i, s]
                val_float = max(0.0, total_new)
                int_part = floor(Int, val_float)
                frac_part = val_float - int_part  


                if step % 1000 == 0 
                    println("Bactérie $i, Espèce $(sym), Step $step, Total_new=$(total_new), Int_part=$(int_part), Frac_part=$(frac_part)")

                end

                target_val = int_part + (rand() < frac_part)
                current_val = get_species(b, sym)
                delta = target_val - current_val

                if abs(delta) > 0.0 
                    add_input!(b, sym, Float64(delta))
                end
                

                history_all[step, i, s] = target_val

            end
        end
    end
end # Fin de la boucle step

println("Simulation terminée.")

# ==============================================================================
# 7. VISUALISATION
# ==============================================================================
times = (1:total_steps) .* dt
colors = [:blue :red :green :cyan] 
labels = ["X", "Y","Y_trans","Z"]

p1 = plot(times, history_all[:, 1, 1:4], title="Bactérie 1 (Source)", label=labels, color=colors, lw=2)

    
p2 = plot(times, history_all[:, 2, 1:4], title="Bactérie 2 (Voisin)", label=labels, color=colors, lw=2)

plot(p1, p2, layout=(1, 2), size=(900, 400))

p3 = plot(times, history_production[:, 1, 2], title="Production de Y de Bactérie 1 (Source)", label=labels, color=colors, lw=2)



println("maximum Y in Bactérie 2: ", maximum(history_all[:, 2, 2]))
println("maximum Y in Bactérie 1: ", maximum(history_all[:, 1, 2]))

println("maximum X in Bactérie 2: ", maximum(history_all[:, 2, 1]))
println("maximum X in Bactérie 1: ", maximum(history_all[:, 1, 1]))

