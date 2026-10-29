"""
Configuración central del proyecto.

Centraliza rutas, hiperparámetros y constantes que antes estaban
duplicadas en los notebooks. Importar este módulo evita rutas duras
y facilita el cambio entre entornos (local / Colab).

Uso:
    from config import CFG
    img_dir = CFG.DATA_DIR / "zr7vgbcyr2-1" / "PAD_UFES_20_ALL_IMAGES"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------
# Raíz del proyecto (el directorio que contiene este archivo).
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    """Constantes del proyecto (inmutable)."""

    # --- Rutas -------------------------------------------------------------
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATA_DIR: Path = PROJECT_ROOT / "Data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "my_logs"
    PREPROCESSED_DIR: Path = PROJECT_ROOT / "preprocessed_multimodal"
    FIGURES_DIR: Path = PROJECT_ROOT / "img_in_exp"

    # --- Dataset principal (Brasil / PAD-UFES-20) --------------------------
    BRAZIL_DIR: Path = DATA_DIR / "zr7vgbcyr2-1"
    BRAZIL_IMAGES_DIR: Path = DATA_DIR / "zr7vgbcyr2-1" / "PAD_UFES_20_ALL_IMAGES"
    BRAZIL_METADATA_CSV: Path = DATA_DIR / "zr7vgbcyr2-1" / "metadata.csv"
    LOCATION_MAPPING_CSV: Path = DATA_DIR / "zr7vgbcyr2-1" / "ubicaciones_unicas.csv"

    # --- Otros datasets ----------------------------------------------------
    ISIC_TRAIN_CSV: Path = DATA_DIR / "ISIC-Data" / "train.csv"
    MIDAS_CSV: Path = DATA_DIR / "midasmultimodalimagedatasetforaibasedskincancer" / "release_midas.csv"

    # --- Artefactos --------------------------------------------------------
    PREPROCESSED_PKL: Path = PREPROCESSED_DIR / "preprocessed_multimodal_data_brasil.pkl"
    BEST_MODEL_KERAS: Path = MODELS_DIR / "modelResNet50TabNet_pro.keras"

    # --- Imágenes ----------------------------------------------------------
    IMG_SHAPE: tuple[int, int, int] = (256, 256, 3)
    BASE_MODEL_INPUT: tuple[int, int, int] = (224, 224, 3)  # entrada de ResNet/Xception
    BATCH_SIZE: int = 64

    # --- Clases de diagnóstico --------------------------------------------
    # Orden alfabético (coincide con LabelEncoder sobre las abreviaturas).
    CLASSES: tuple[str, ...] = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
    NUM_CLASSES: int = 6

    # --- Escala Fitzpatrick de interés ------------------------------------
    FITZPATRICK_RANGE: tuple[float, ...] = (3.0, 4.0, 5.0, 6.0)

    # --- Hiperparámetros de entrenamiento ---------------------------------
    LEARNING_RATE: float = 1e-4
    FINE_TUNE_LR: float = 5e-5
    EPOCHS: int = 100
    EARLY_STOPPING_PATIENCE: int = 20
    DROPOUT: float = 0.4

    # --- Semilla de reproducibilidad --------------------------------------
    SEED: int = 42

    # --- Features tabulares (29) ------------------------------------------
    TABULAR_FEATURES: tuple[str, ...] = (
        "smoke", "drink", "background_father", "background_mother", "age",
        "pesticide", "gender", "skin_cancer_history", "cancer_history",
        "has_piped_water", "has_sewage_system", "fitspatrick", "region",
        "diameter_1", "diameter_2", "itch", "grew", "hurt",
        "changed", "bleed", "elevation",
    )

    def __post_init__(self) -> None:
        """Crea los directorios que no existan (idempotente)."""
        for d in (self.MODELS_DIR, self.LOGS_DIR, self.PREPROCESSED_DIR, self.FIGURES_DIR):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
