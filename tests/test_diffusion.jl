include("../src/BioKan.jl")
using .BioKan
using Catalyst, JumpProcesses, Distributions, Plots

# ==============================================================================
# 1. PARAMÉTRAGE DE L'UNIVERS
# ==============================================================================
n_bacteries = 2
taille_espace = 5.0   # Un peu plus d'espace pour voir la diffusion
distance_comm = 10.0  # Rayon de connexion
n_segments_diffusion = 5
dt = 0.1
total_steps = Int(10.0 / dt)

# ==============================================================================
# 2. CONSTRUCTION DU RÉSEAU
# ==============================================================================
net = BioNetwork(distance_comm, n_segments_diffusion)

# Placement manuel pour tester la distance (Source à 0, Récepteur à 2.0)
b1 = Bacterium(1, 1.0, 1.0)
b2 = Bacterium(2, 3.0, 1.0) # Distance = 2.0

add_bacterium!(net, b1)
add_bacterium!(net, b2)

build_edges!(net)
println("Réseau : $(length(net.nodes)) bactéries, $(length(net.edges)) edges.")
println("Distance entre B1 et B2 : $(net.edges[2]) (doit être proche de $distance_comm)")

# ==============================================================================
# 3. DÉFINITION DE LA PHYSIQUE (BIO-RÉALISTE)
# ==============================================================================
species_names = [:, :B, :C, :G_on, :G_off] 

# --- A. CONSTANTES DE DIFFUSION (D) ---
# D determine la vitesse de propagation.
# A et C sont des molécules de signalisation (ex: AHL) -> D élevé.
# B, G_on, G_off sont internes (Protéines, ADN) -> D = 0 (Confinement total).
D_dict = Dict(
    :A => 0.5,    # Signal d'entrée (Rapide)
    :B => 0.0,     # Inhibiteur INTERNE (Doit rester pour agir sur l'ADN)
    :C => 0.5,     # Signal de Sortie (Diffuse vers les autres couches)
    :G_on => 0.0,  # ADN (Fixe)
    :G_off => 0.0  # ADN (Fixe)
)

# --- B. CONSTANTES DE DÉGRADATION (Gamma) ---
# Gamma détermine la durée de vie dans le milieu ET dans la cellule.
# B doit disparaître vite pour que le "Pulse" puisse redescendre.
gamma_dict = Dict(
    :A => 0.3,     # Stable
    :B => 0.8,     # Instable (Essentiel pour le pulse !)
    :C => 0.3,     # Stable
    :G_on => 0.0,  # Immortel
    :G_off => 0.0  # Immortel
)
T_max_kernel = 100.0   # Mémoire tampon de diffusion
R_cell = 0.5          # Rayon physique de la bactérie (pour l'émission)

# ==============================================================================
# 4. INITIALISATION SOLVER & CIRCUIT
# ==============================================================================
circuit = create_iFFL_circuit(:node_generic)

# Paramètres du circuit génétique (Interne)
# k_prod_y(B), k_prod_z(C), k_inh, k_rel, k_deg

p_vals = [50.0, 100.0, 10.0, 1.0, 0.1] 
# Note: Production très forte (50, 100) pour contrer la dilution numérique initiale

tspan = (0.0, 10000.0) 
integrators = []

# B1 est la source : On lui donne beaucoup de A (Input) au début
u0_source = [1000, 0, 0, 1, 0] 
u0_receiver = [0, 0, 0, 1, 0]


for i in 1:n_bacteries
    u0 = (i == 1) ? u0_source : u0_receiver
    prob = DiscreteProblem(circuit, Int.(u0), tspan, p_vals)
    # save_positions=(false, false) accélère énormément le JumpProcess
    jump_prob = JumpProblem(circuit, prob, Direct(), save_positions=(false, false))
    push!(integrators, init(jump_prob, SSAStepper()))
end

# ==============================================================================
# 5. CALCUL ROBUSTE DES KERNELS (PHYSIQUE)
# ==============================================================================


kernels, t_vec, n_species = compute_diffusion_kernels_physics(
    net.edges, D_dict, gamma_dict, species_names, dt, T_max_kernel, R_cell
)

# ==============================================================================
# 6. BOUCLE PRINCIPALE
# ==============================================================================
# Buffer circulaire pour la convolution
buffer = zeros(Float64, n_bacteries, n_species, length(t_vec))
current_ptr = 1

# Extraction des facteurs de rétention "physiques" depuis les kernels calculés
# Cela garantit que la boucle respecte exactement D et Gamma
retention_factors = zeros(n_species)
if haskey(kernels, (1, 1))
    # Ce qui reste au temps t=dt
    retention_factors = kernels[(1, 1)][:, 1] 
else
    error("Pas de self-loop trouvée ! Le réseau est mal construit.")
end





history_all = zeros(total_steps, n_bacteries, n_species)
println("Simulation : Démarrage...")

for step in 1:total_steps
    t_sim = step * dt

    # --- A. BIOLOGIE (Odes) ---
    for i in 1:n_bacteries
        step!(integrators[i], dt, true)
    end

    # --- B. PHYSIQUE (Séparation Interne / Externe) ---
    flux_emissions = zeros(n_bacteries, n_species)
    retained_stock = zeros(n_bacteries, n_species)

    for i in 1:n_bacteries
        for s in 1:n_species
            qty = max(0.0, integrators[i].u[s]) # Sécurité
            
            # Si D=0 (B, Gènes), retention_factors[s] est proche de 1.0 (juste dégradation)
            # Si D grand (A, C), retention_factors[s] est petit (ça fuit)
            
            retained = qty * retention_factors[s]
            leaked   = qty * (1.0 - retention_factors[s])
            
            retained_stock[i, s] = retained
            flux_emissions[i, s] = leaked
        end
    end

    # --- C. TRANSPORT (Convolution) ---
    received, next_ptr = update_diffusion!(buffer, kernels, flux_emissions, current_ptr)
    current_ptr = next_ptr

    # --- D. MISE À JOUR ---
    for i in 1:n_bacteries
        integ = integrators[i] # On récupère l'intégrateur
        
        for s in 1:n_species
            # Bilan de masse
            total_new = retained_stock[i, s] + received[i, s]
            
            # Conversion Float -> Int pour SSA
            # ASTUCE : Pour éviter de perdre les "0.9" molécules, on peut faire un arrondi probabiliste
            # Mais floor() est plus stable pour commencer.
            val_int = floor(Int, max(0.0, total_new))
            
            
            # Injection manuelle dans l'état du solver
            integ.u[s] = val_int
            # Historique pour le plot
            history_all[step, i, s] = val_int
        end


        reset_aggregated_jumps!(integ)

    end
end


println("Simulation terminée.")

# ==============================================================================
# 7. PLOT
# ==============================================================================
times = (1:total_steps) .* dt


colors = [:blue :red :green] 
labels = ["A (Input)" "B (Inhibiteur)" "C (Output)"]

p1 = plot(times, history_all[:, 1, 1:3], title="Bactérie 1 (Source)", label=labels, color=colors, lw=2)
p2 = plot(times, history_all[:, 2, 1:3], title="Bactérie 2 (Voisin)", label=labels, color=colors, lw=2)
plot(p1, p2, layout=(2,1), size=(800,600))
