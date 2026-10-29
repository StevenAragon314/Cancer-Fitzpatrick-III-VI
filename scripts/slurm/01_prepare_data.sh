#!/bin/bash
#===============================================================================
# SLURM · Etapa 1 — Preparación de datos (CPU)
#
# Ejecuta los notebooks 01 (adquisición/estandarización) y 02 (limpieza y
# preprocesamiento) y genera el artefacto preprocesado que consumen las
# etapas de entrenamiento y evaluación.
#
# Envío:  sbatch scripts/slurm/01_prepare_data.sh
#
# AJUSTA las directivas #SBATCH a tu cluster (partición, memoria, tiempo).
#===============================================================================
#SBATCH --job-name=skin-cancer-data
#SBATCH --partition=cpu              # <-- partición CPU de tu HPC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#--- Configuración -------------------------------------------------------------
CONDA_ENV="skin-cancer"              # nombre del entorno conda del proyecto
# Ruta al conda del HPC (ajusta a tu instalación):
CONDA_ROOT="/path/to/miniconda3"

# Raíz del proyecto (dos niveles arriba de scripts/slurm/).
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

#--- Entorno -------------------------------------------------------------------
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

cd "${PROJECT_ROOT}"
mkdir -p logs

echo "=== Etapa 1: Preparación de datos ==="
echo "Proyecto: ${PROJECT_ROOT}"
echo "Entorno:  ${CONDA_ENV}"
echo "Inicio:   $(date)"

#--- Ejecución de notebooks ----------------------------------------------------
# 01: adquisición y estandarización de los 3 datasets
jupyter nbconvert --to notebook --execute --inplace Code/01_getting_data.ipynb

# 02: limpieza, split y preprocesamiento (genera el .pkl)
jupyter nbconvert --to notebook --execute --inplace Code/02_data_cleaning_brasil.ipynb

echo "=== Etapa 1 completada: $(date) ==="
echo "Artefacto: preprocessed_multimodal/preprocessed_multimodal_data_brasil.pkl"
