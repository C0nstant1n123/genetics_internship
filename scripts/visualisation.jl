using JLD2
using Plots
using Printf
using Statistics
using Dates

# ==============================================================================
# 🛠️ PATCH DE COMPATIBILITÉ JLD2 (VERSION V4 - rconvert)
# ==============================================================================
import JLD2

# --- 1. GESTION CRITIQUE DES TYPES INTERNES (TypeVar) ---
# C'est ce qui causait l'erreur "TypeError: in Tuple... expected Type"
# On intercepte la reconstruction de TypeVar pour convertir le nom (String -> Symbol)
function JLD2.rconvert(::Type{TypeVar}, x::JLD2.ReconstructedMutable{:TypeVar})
    # x.name est une String dans le fichier, on le force en Symbol
    return TypeVar(Symbol(x.name), x.lb, x.ub)
end

# --- 2. GESTION DES PAIRS (String -> Symbol) ---

# Pair{Symbol, Any}
function JLD2.rconvert(::Type{Pair{Symbol, Any}}, x::JLD2.ReconstructedMutable{Symbol("Pair{Symbol, Any}")})
    return Pair{Symbol, Any}(Symbol(x.first), x.second)
end

# Pair{Symbol, Real}
function JLD2.rconvert(::Type{Pair{Symbol, Real}}, x::JLD2.ReconstructedMutable{Symbol("Pair{Symbol, Real}")})
    return Pair{Symbol, Real}(Symbol(x.first), x.second)
end

# Pair{Symbol, Float64}
function JLD2.rconvert(::Type{Pair{Symbol, Float64}}, x::JLD2.ReconstructedMutable{Symbol("Pair{Symbol, Float64}")})
    return Pair{Symbol, Float64}(Symbol(x.first), x.second)
end

# Pair{Symbol, Dict}
function JLD2.rconvert(::Type{Pair{Symbol, Dict{Symbol, Float64}}}, x::JLD2.ReconstructedMutable{Symbol("Pair{Symbol, Dict{Symbol, Float64}}")})
    return Pair{Symbol, Dict{Symbol, Float64}}(Symbol(x.first), x.second)
end

# --- 3. FILET DE SÉCURITÉ ---
# Si un Pair{Symbol, Any} arrive avec un Dict précis, on l'accepte
function JLD2.rconvert(::Type{Pair{Symbol, Any}}, p::Pair{Symbol, <:Any})
    return Pair{Symbol, Any}(p.first, p.second)
end
# ==============================================================================




# ==============================================================================
# 1. CONFIGURATION & SCAN AUTO
# ==============================================================================

# Adapte ce chemin si nécessaire
const BATCH_ROOT = "/Users/constantindeumier/Desktop/BioKan_project/outputs/batch_03"

function scan_simulations(root_dir)
    if !isdir(root_dir)
        println("⚠️ ATTENTION : Le dossier $root_dir n'existe pas.")
        return String[]
    end

    paths = String[]
    println("🔍 Scan du dossier : $root_dir ...")
    
    for (root, dirs, files) in walkdir(root_dir)
        for file in files
            if endswith(file, ".jld2")
                push!(paths, joinpath(root, file))
            end
        end
    end
    
    println("✅ Trouvé $(length(paths)) fichiers de simulation.")
    return paths
end

const ALL_SIM_PATHS = scan_simulations(BATCH_ROOT)

# ==============================================================================
# 2. CHARGEMENT
# ==============================================================================

function load_sim(index::Int)
    if index < 1 || index > length(ALL_SIM_PATHS)
        error("Index invalide.")
    end
    path = ALL_SIM_PATHS[index]
    println("📂 Chargement [$index] : $(basename(dirname(path)))/$(basename(path))")
    return load_object(path)
end

function load_sim(keyword::String, replica::Int)
    filename = @sprintf("sim_%d.jld2", replica)
    candidates = filter(p -> occursin(keyword, p) && endswith(p, filename), ALL_SIM_PATHS)

    if isempty(candidates)
        error("❌ Aucune sim trouvée pour '$keyword' (replica $replica)")
    end
    
    path = first(candidates)
    println("📂 Chargement spécifique : $(basename(dirname(path)))/$(basename(path))")
    return load_object(path)
end

# ==============================================================================
# 3. VISUALISATION (CORRECTION CLÉS STRING)
# ==============================================================================

function list_species(data)
    # CORRECTION ICI : "config" au lieu de :config
    config = data["config"] 

    # On gère si la config interne utilise des Symboles ou Strings
    if haskey(config, :species_names)
        names = config[:species_names]
    elseif haskey(config, "species_names")
        names = config["species_names"]
    else
        # Fallback
        n_species = size(data["data"], 3)
        names = ["Espèce $i" for i in 1:n_species]
    end

    println("\n===  ESPÈCES DISPONIBLES ===")
    for (i, name) in enumerate(names)
        println("  [$i] : $name")
    end
    println("==============================\n")
    return names
end

function plot_sim(data; bacteria=[1], species=nothing, t_plot=nothing)
    # CORRECTION ICI : Accès par String
    history = data["data"] 
    t_vec   = data["time"]    
    config  = data["config"]

    # --- FILTRAGE TEMPOREL ---
    if !isnothing(t_plot)
        t_start, t_end = t_plot
        indices = findall(x -> t_start <= x <= t_end, t_vec)
        if isempty(indices)
            return plot(title="Plage temporelle vide")
        end
        t_vec = t_vec[indices]
        history = history[indices, :, :]
    end

    # --- GESTION DES NOMS ---
    if haskey(config, :species_names)
        raw_names = config[:species_names]
    elseif haskey(config, "species_names")
        raw_names = config["species_names"]
    else
        raw_names = ["Esp $i" for i in 1:size(history, 3)]
    end
    all_names = string.(raw_names)

    # --- SÉLECTION ---
    if isnothing(species)
        species_indices = 1:length(all_names)
    elseif species isa Vector{<:Integer}
        species_indices = species
    else
        # Recherche par nom
        s_str = string.(species)
        species_indices = [findfirst(==(n), all_names) for n in s_str]
    end

    if any(isnothing, species_indices)
        error("Espèce introuvable dans : $all_names")
    end

    # --- PLOT ---
    n_plots = length(bacteria)
    colors = palette(:tab10) 

    p = plot(layout = (n_plots, 1), 
             size   = (800, 350 * n_plots),
             xlabel = "Temps (s)",
             margin = 5Plots.mm)

    for (i, b_idx) in enumerate(bacteria)
        title!(p, subplot=i, "Bactérie $b_idx")
        ylabel!(p, subplot=i, "Conc. (nM)")

        for (k, s_idx) in enumerate(species_indices)
            series = history[:, b_idx, s_idx] 
            
            # Calcul CV simple
            mu = mean(series)
            CV = (mu > 1e-9) ? std(series)/mu : 0.0
            
            label_str = "$(all_names[s_idx]) (CV: $(round(CV, digits=2)))"

            plot!(p, t_vec, series,
                  subplot = i,
                  label   = label_str,
                  lw      = 2,
                  color   = colors[mod1(k, length(colors))])
        end
    end

    display(p)
    return p
end

function show_sweep_infos(data)
    # CORRECTION ICI
    config = data["config"]
    
    # Gestion souple id
    id = get(config, :replica_id, get(config, "replica_id", "?"))
    
    println("\n=== 🔎 INFO SIMULATION (ID: $id) ===")

    # Affichage param bio (souvent c'est des Symbol dans config)
    # On essaie d'accéder aux clés proprement
    println("📌 Paramètres :")
    
    for k in [:K_output, :K_input, :k_sec, :k_transl, :v_max]
        if haskey(config, k)
            println("   ▶ $k = $(config[k])")
        elseif haskey(config, string(k))
             println("   ▶ $k = $(config[string(k)])")
        end
    end
    println("================================\n")
end

# ==============================================================================
# 4. TEST RAPIDE
# ==============================================================================
if !isempty(ALL_SIM_PATHS)
    println("--- TEST AUTOMATIQUE ---")
    d = load_sim("k_transl=5.2_v_max=0.1",1)     # Charge le premier fichier
    show_sweep_infos(d) # Affiche les infos
    list_species(d)     # Liste les molécules
    # plot_sim(d, bacteria=[1], species=[1]) # Décommenter pour voir le plot
end


plot_sim(d, bacteria=[1], species=[1,2,4,5], t_plot=nothing)

show_sweep_infos(load_sim("k_transl=5.2_v_max=0.1",1))