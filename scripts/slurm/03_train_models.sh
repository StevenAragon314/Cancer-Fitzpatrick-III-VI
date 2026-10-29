#!/bin/bash
#===============================================================================
# SLURM · Etapa 2 — Entrenamiento de modelos (GPU)
#
# Ejecuta el notebook 03 (desarrollo de modelos multimodales). Requiere GPU.
#
# Envío:  sbatch scripts/slurm/03_train_models.sh
#
# AJUSTA las directivas #SBATCH a tu cluster (partición GPU, nº de GPUs,
# memoria, tiempo). El entrenamiento es la etapa más costosa.
#===============================================================================
#SBATCH --job-name=skin-cancer-train
#SBATCH --partition=gpu              # <-- partición GPU de tu HPC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1                 # <-- 1 GPU (ajusta si usas más)
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#--- Configuración -------------------------------------------------------------
CONDA_ENV="skin-cancer"
CONDA_ROOT="/path/to/miniconda3"     # <-- ajusta a tu instalación

# Raíz del proyecto.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

#--- Entorno -------------------------------------------------------------------
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# Variables de CUDA (si tu HPC usa módulos, descomenta y ajusta):
# module load cuda/12.x
# module load gcc/12.x

cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== Etapa 2: Entrenamiento de modelos ==="
echo "Proyecto: ${PROJECT_ROOT}"
echo "GPU(s):   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'n/d')"
echo "Inicio:   $(date)"

#--- Ejecución -----------------------------------------------------------------
# 03: desarrollo y entrenamiento de las arquitecturas multimodales
jupyter nbconvert --to notebook --execute --inplace Code/03_developing_models.ipynb

echo "=== Etapa 2 completada: $(date) ==="
echo "Modelos guardados en: models/"
