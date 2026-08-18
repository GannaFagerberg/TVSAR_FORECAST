#!/bin/bash -l

#SBATCH -A naiss2026-4-20      # specifies	YOUR_NAISS_PROJECT
#SBATCH -J AR1
#SBATCH -t 4:00:00
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH --array=1-5

#SBATCH --mail-user=ganna.fagerberg@stat.su.se
#SBATCH --mail-type=BEGIN,END,FAIL

# Save SLURM output
#SBATCH -o /home/x_ganfa/slurm_logs/%A_%a.out

# Load Julia
module load julia/1.10.2-bdist

# Ensure log directory exists
mkdir -p /home/x_ganfa/slurm_logs

# (Optional) Pass maximum array index to Julia
#export MAX_ID=5

# Run Julia using the precompiled environment
julia --project=/home/x_ganfa/julia-env ~/poisson_sar_ref.jl