#!/bin/bash
#===============================================================================
# SLURM · Etapa 3 — Evaluación e interpretabilidad (GPU)
#
# Ejecuta el notebook 04 (evaluación del mejor modelo, matriz de confusión,
# análisis de error por Fitzpatrick, Grad-CAM y máscaras de TabNet).
#
# Envío:  sbatch scripts/slurm/04_evaluate.sh
#
# AJUSTA las directivas #SBATCH a tu cluster.
#===============================================================================
#SBATCH --job-name=skin-cancer-eval
#SBATCH --partition=gpu              # <-- partición GPU de tu HPC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
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

cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== Etapa 3: Evaluación e interpretabilidad ==="
echo "Proyecto: ${PROJECT_ROOT}"
echo "Inicio:   $(date)"

#--- Ejecución -----------------------------------------------------------------
# 04: evaluación del modelo ganador y generación de explicaciones
jupyter nbconvert --to notebook --execute --inplace Code/04_best_models_evaluation.ipynb

echo "=== Etapa 3 completada: $(date) ==="
echo "Figuras de resultados en: img_in_exp/"
