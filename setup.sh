#!/bin/bash
#SBATCH --job-name=build_env      # Nom du job
#SBATCH --output=build_env.log    # Log de sortie
#SBATCH --ntasks=1                # 1 tache
#SBATCH --cpus-per-task=2         # 2 CPUs pour compiler vite
#SBATCH --mem=4G                  # 4 Go de RAM suffisent
#SBATCH --time=00:20:00           # 20 minutes max

echo "--- Début de la construction de l'environnement ---"

# 1. Charger le module (Version fixe pour la stabilité)
module load Julia/1.10.5-linux-x86_64

# 2. Construction de l'environnement
# - Pkg.develop : Enregistre ta lib locale (chemin relatif)
# - Pkg.instantiate : Télécharge les paquets manquants
# - Pkg.precompile : Compile tout maintenant (pour ne pas le faire pendant tes calculs)
julia --project=. -e 'using Pkg; Pkg.develop(path="lib/rate-distortion-example"); Pkg.instantiate(); Pkg.precompile()'

echo "--- Environnement construit et compilé avec succès ---"
