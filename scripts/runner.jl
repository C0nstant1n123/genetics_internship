using Distributed
using Pkg
using JLD2
using Printf

# ==============================================================================
# 1. INITIALISATION & PATHS (Inspiré de ton script qui marche)
# ==============================================================================

# On remonte d'un niveau (scripts/ -> racine)
PROJECT_ROOT = dirname(@__DIR__) 
Pkg.activate(PROJECT_ROOT)

println("📍 Racine du projet : $PROJECT_ROOT")

# --- CHEMINS ABSOLUS (CRUCIAL POUR LES WORKERS) ---
PATH_BIOKAN = joinpath(PROJECT_ROOT, "src", "BioKan.jl")

# ATTENTION : Vérifie ici où est ton fichier de simulation. 
# Dans ton script qui marche, c'était "scripts/simulation.jl".
# Si tu l'as bougé dans src/, change cette ligne.
PATH_BIOSIM = joinpath(PROJECT_ROOT, "scripts", "simulation.jl") 

# --- I/O POUR LE BATCH ORGANISÉ ---
INPUT_FILE  = joinpath(PROJECT_ROOT, "inputs", "experiment_batch_04.jld2")
OUTPUT_ROOT = joinpath(PROJECT_ROOT, "outputs", "batch_04")

# ==============================================================================
# 2. DÉMARRAGE DES WORKERS
# ==============================================================================

if nprocs() == 1
    if haskey(ENV, "SLURM_CPUS_PER_TASK")
        num_cores = parse(Int, ENV["SLURM_CPUS_PER_TASK"])
        println("🏢 Cluster : Ajout de $(num_cores - 1) workers.")
        addprocs(num_cores - 1)
    else
        println("🏠 Local : Ajout de 4 workers.")
        addprocs(4) 
    end
end

println("🚀 Processus actifs : $(nprocs())")

# ==============================================================================
# 3. CHARGEMENT SUR LES WORKERS (La partie "Robuste")
# ==============================================================================

println("📂 Chargement du code sur les workers...")

@everywhere begin
    using Pkg
    using Distributed
    # Les workers activent l'environnement
    Pkg.activate($PROJECT_ROOT) 
    
    using JLD2, Dates, Printf, Random
    
    # On charge Catalyst/MTK ici pour être sûr
    using Catalyst, ModelingToolkit, DifferentialEquations
end

# Inclusion via chemins absolus interpolés ($) -> C'est ça qui corrige l'erreur !
@everywhere include($PATH_BIOKAN)
@everywhere include($PATH_BIOSIM)

@everywhere using .BioKan
@everywhere using .BioSim 

# ==============================================================================
# 4. TÂCHE WORKER (La partie "Organisée")
# ==============================================================================

@everywhere function worker_task(config, root_output_dir)
    id = config[:id]
    rep_id = config[:replica_id]
    
    # --- LOGIQUE DE CLASSEMENT ---
    # 1. On récupère le nom du dossier généré par generate_configs_3.jl
    subfolder = get(config, :folder_name, "misc")
    
    # 2. On construit le chemin du sous-dossier
    target_dir = joinpath(root_output_dir, subfolder)
    
    # 3. On crée le dossier (Thread-safe : si plusieurs essaient, ça ne plante pas)
    mkpath(target_dir)

    # 4. Nom du fichier : sim_{id_de_repetition}.jld2
    # Plus besoin de l'ID global complexe car on est dans un dossier spécifique
    filename = "sim_$(rep_id).jld2"
    filepath = joinpath(target_dir, filename)

    # Skip si existe
    if isfile(filepath)
        return "SKIP: $subfolder/$filename"
    end

    try
        # Exécution
        # history est l'objet complexe (ODESolution), time_vector est le vecteur temps
        raw_sol, time_vector = BioSim.run_single_simulation(config)

        # --- ASSAINISSEMENT DES DONNÉES (SANITIZING) ---
        
        # 1. On convertit la solution complexe en simple Matrice de nombres
        # Array(sol) transforme l'objet ODESolution en Array{Float64, 2}
        # C'est universel et compatible partout.
        clean_matrix = Array(raw_sol) 

        # 2. On nettoie la config (pour virer les Symboles ou types bizarres)
        clean_config = Dict{String, Any}()
        for (k, v) in config
            # On garde les nombres et strings tels quels
            if v isa Number || v isa String
                clean_config[String(k)] = v
            else
                # Le reste (Symboles, Enums...), on le convertit en String pour être sûr
                clean_config[String(k)] = string(v)
            end
        end

        # 3. Sauvegarde Robuste
        # On ne sauvegarde QUE des structures primitives (Dict, Array, String, Float64)
        save_object(filepath, Dict(
            "config"  => clean_config,
            "data"    => clean_matrix,  # On renomme 'history' en 'data' pour être clair
            "time"    => time_vector
        ))

        return "OK: $subfolder/$filename"
    catch e
        return "ERROR: $subfolder -> $e"
    end
end

# ==============================================================================
# 5. MAIN
# ==============================================================================

function main()
    # Création racine du batch
    if !isdir(OUTPUT_ROOT)
        mkpath(OUTPUT_ROOT)
    end

    if !isfile(INPUT_FILE)
        println("❌ Input introuvable : $INPUT_FILE")
        println("👉 As-tu lancé scripts/generate_configs_3.jl ?")
        return
    end

    tasks = load_object(INPUT_FILE)
    println("🔥 Lancement de $(length(tasks)) simulations...")
    println("📂 Sortie : $OUTPUT_ROOT")

    # Lancement parallèle
    results = pmap(t -> worker_task(t, OUTPUT_ROOT), tasks)

    # Petit rapport
    fails = count(r -> startswith(r, "ERROR"), results)
    oks = count(r -> startswith(r, "OK"), results)
    
    println("\n✅ Terminé.")
    println("   SUCCESS : $oks")
    println("   FAIL    : $fails")
    
    if fails > 0
        println("⚠️ Exemple d'erreur :")
        println(first(filter(r -> startswith(r, "ERROR"), results)))
    end
end

main()
