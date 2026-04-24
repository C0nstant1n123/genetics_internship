#!/bin/bash
# ==============================================================================
# lancer-optim-hebbian.sh — CMA-ES Hebbian sur cluster SLURM (multi-nœuds)
# ==============================================================================
#
# Architecture :
#   2 nœuds × 7 workers × 4 threads = 56 CPUs
#   λ = 14 (candidats CMA-ES/génération) → 1 génération = 1 round de 14 simulations
#
# Dimensionnement RAM :
#   RAM/worker : ~1.2 GB (Julia/Catalyst) + ~0.6 GB (SSA 27 bactéries) = ~2 GB
#   CPUs/nœud : 7 workers × 4 threads = 28 CPUs → 2 GB/worker = 500 MB/CPU < 4000 MB/CPU ✓
#   (+1 CPU pour le maître sur le nœud 1)
#
# Usage :
#   sbatch lancer-optim-hebbian.sh
# ==============================================================================

#SBATCH --job-name=BioKan_CMA_Hebbian
#SBATCH --partition=generic
#SBATCH --nodes=1
#SBATCH --exclude=drago31010021
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14          # 1 thread par candidat CMA-ES (λ=14)
#SBATCH --mem-per-cpu=2000M         # 2 GB/CPU, bien sous la limite 4000 MB/CPU
#SBATCH --time=15-00:00:00          # partition generic : 15 jours max
#SBATCH --output=logs/optim_hebbian_%j.out
#SBATCH --error=logs/optim_hebbian_%j.err

cd $SLURM_SUBMIT_DIR
module load Julia/1.10.5-linux-x86_64
mkdir -p outputs/optim_hebbian logs

echo "=== CMA-ES Hebbian (multi-nœuds) ==="
echo "Nœuds alloués : $SLURM_JOB_NUM_NODES"
echo "Machines      : $SLURM_JOB_NODELIST"
echo "CPUs/nœud     : $SLURM_CPUS_PER_TASK"
echo "RAM/CPU       : $SLURM_MEM_PER_CPU MB"
echo "Job ID        : $SLURM_JOB_ID"
echo "Début         : $(date)"

# Étape 1 — précompilation packages + modèle Catalyst avec 1 thread
julia --project=/lustre/home/cnb/agoni/BioKan_project --threads=14 scripts/optim_hebbian.jl 2>&1 | tee -a logs/optim_hebbian_live.out

echo "Fin : $(date)"
echo "=== FIN DU JOB ==="
