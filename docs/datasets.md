# Datasets

Los datos **no se incluyen** en el repositorio por su tamaño (decenas de miles de imágenes) y por las restricciones de uso de cada fuente. Este documento describe cómo obtenerlos y la estructura esperada.

## Estructura esperada

```
Data/
├── ISIC-Data/
│   └── train.csv
├── midasmultimodalimagedatasetforaibasedskincancer/
│   └── release_midas.csv
└── zr7vgbcyr2-1/
    ├── metadata.csv
    ├── ubicaciones_unicas.csv
    └── PAD_UFES_20_ALL_IMAGES/
        ├── PAT_001_001_001.png
        └── ...
```

## 1. ISIC 2020 (International Skin Imaging Collaboration)

- **Fuente:** [ISIC Archive](https://www.isic-archive.com/)
- **Contenido:** ~33.000 imágenes dermatoscópicas con metadatos (edad aproximada, sexo, sitio anatómico, diagnóstico).
- **Uso en el proyecto:** se extraen los **melanomas en palmas/plantas** (`anatom_site_general_challenge == "palms/soles"`) como *proxy* de lesiones en piel oscura, ya que ISIC no incluye la escala Fitzpatrick.
- **Archivo:** `Data/ISIC-Data/train.csv`

## 2. MIDAS (Stanford)

- **Fuente:** *Multimodal Image Dataset for AI-based Skin Cancer* (MIDAS), disponible en [Kaggle](https://www.kaggle.com/).
- **Contenido:** ~10.000 imágenes con metadatos clínicos ricos, incluyendo la **escala Fitzpatrick explícita** (`midas_fitzpatrick`).
- **Uso en el proyecto:** se filtran los niveles `iii darker white skin` a `vi dark brown or black skin` (Fitzpatrick III–VI).
- **Archivo:** `Data/midasmultimodalimagedatasetforaibasedskincancer/release_midas.csv` (separador `;`).

## 3. PAD-UFES-20 (Brasil) — **dataset principal**

- **Fuente:** Universidad Federal de Espírito Santo (UFES), disponible en [Kaggle](https://www.kaggle.com/datasets/).
- **Contenido:** ~2.300 imágenes dermatoscópicas de pacientes brasileños con metadatos clínicos detallados (antecedentes familiares, síntomas, factores de riesgo, escala Fitzpatrick).
- **Uso en el proyecto:** es el dataset de **entrenamiento y evaluación principal**, filtrado a Fitzpatrick III–VI.
- **Archivos:**
  - `Data/zr7vgbcyr2-1/metadata.csv`
  - `Data/zr7vgbcyr2-1/ubicaciones_unicas.csv` (mapeo de localizaciones)
  - `Data/zr7vgbcyr2-1/PAD_UFES_20_ALL_IMAGES/` (imágenes PNG)

## Esquema estandarizado

Tras la estandarización (notebook `01_getting_data.ipynb`), cada dataset se convierte a un esquema común de 8 columnas:

| Columna | Descripción |
|---|---|
| `image_id` | Nombre del archivo de imagen |
| `patient_id` | Identificador del paciente |
| `age` | Edad |
| `sex` | Sexo (`male`/`female`) |
| `location` | Región anatómica estandarizada |
| `diagnosis` | Diagnóstico abreviado (`ACK`, `BCC`, `MEL`, `NEV`, `SCC`, `SEK`, `otro`) |
| `fitzpatrick_skin_type` | Escala Fitzpatrick (0 = desconocido) |
| `source` | Origen del dataset (`isic`, `stanford`, `brasil`) |

## Clases de diagnóstico

| Abreviatura | Diagnóstico |
|---|---|
| `ACK` | Queratosis actínica (maligna) |
| `BCC` | Carcinoma basocelular |
| `MEL` | Melanoma |
| `NEV` | Nevo melanocítico (benigno) |
| `SCC` | Carcinoma espinocelular |
| `SEK` | Queratosis seborreica (benigna) |
