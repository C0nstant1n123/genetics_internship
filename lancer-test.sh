#!/bin/bash
#SBATCH --job-name=BioKan_Batch03
#SBATCH --partition=generic
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # 1 Processus Maître (Julia)
#SBATCH --cpus-per-task=20         # 20 Workers (C'est un bon équilibre)
#SBATCH --mem=40G                  # 40 Go Total (Soit 2 Go par worker)
#SBATCH --time=10:00:00            # Temps max alloué
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err


cd $SLURM_SUBMIT_DIR
module load Julia/1.10.5-linux-x86_64
mkdir -p inputs outputs logs

echo "=== DÉBUT DU JOB ==="
echo "Machine : $SLURMD_NODENAME"
# C'est cette variable qui dira à Julia combien de workers lancer
echo "CPUs alloués : $SLURM_CPUS_PER_TASK" 

# 1. Génération
echo "--> [1/2] Génération des configurations..."
julia --project=. scripts/generate_configs_3.jl || exit 1

# 2. Lancement
echo "--> [2/2] Lancement du Runner..."
# Note bien le --project=. qui force l'utilisation de la racine
julia --project=. scripts/runner.jl

echo "=== FIN DU JOB ==="

