#!/bin/bash -l

#SBATCH -A naiss2026-4-20
#SBATCH -J TVSAR_collect
#SBATCH -t 0:30:00
#SBATCH -c 1
#SBATCH --mem=8G

#SBATCH --mail-user=ganna.fagerberg@stat.su.se
#SBATCH --mail-type=END,FAIL

#SBATCH -o /home/x_ganfa/slurm_logs/%A_collect.out


############################################################
# 1. Load Julia and R
############################################################

module load julia/1.10.2-bdist
module load R/4.4.0-hpc1-gcc-11.3.0-bare

export R_HOME="$(R RHOME)"
export LD_LIBRARY_PATH="${R_HOME}/lib:${LD_LIBRARY_PATH}"

# Keep this because this is where you installed forecast
export R_LIBS_USER="/home/x_ganfa/R/4.4.0-library"


############################################################
# 2. Locations
############################################################

PROJECT_DIR="/home/x_ganfa/TVSAR_FORECAST"

SCRIPT="${PROJECT_DIR}/cluster/scripts/choose_model_LPS.jl"

DATA_FILE="/home/x_ganfa/AustralianElecPriceDemand202512_hourly.csv"

RESULTS_DIR="/home/x_ganfa/TVSAR_results"


############################################################
# 3. COLLECT
############################################################

export TVSAR_ACTION="collect"
export TVSAR_RUN_SIZE="mini"

export TVSAR_DATA_FILE="${DATA_FILE}"
export TVSAR_RESULTS_DIR="${RESULTS_DIR}"


############################################################
# 4. Run collector once
############################################################

julia \
    --project="${PROJECT_DIR}" \
    "${SCRIPT}"