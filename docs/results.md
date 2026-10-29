# Resultados

Resumen de los resultados obtenidos en la evaluación del sistema multimodal de detección de cáncer de piel.

## Modelo ganador

**ResNet50 + TabNet** (`models/modelResNet50TabNet_pro.keras`)

- **Rama de visión:** ResNet50 preentrenada en ImageNet (congelada en la fase inicial, fine-tuning de las últimas capas en la fase final).
- **Rama tabular:** TabNet encoder (atención sobre las 29 features clínicas).
- **Fusión:** concatenación de ambas ramas + cabezal denso (256 unidades, dropout 0.4) + salida softmax de 6 clases.

## Métricas principales

| Métrica | Entrenamiento | Validación |
|---|---|---|
| F1 macro | ~0.73 | ~0.72 |
| AUC | ~0.92 | ~0.89 |
| Accuracy | ~0.67 | ~0.70 |
| Loss | ~0.49 | ~1.07 |

> Valores correspondientes al mejor checkpoint (monitoreo por `val_f1_score_macro`). Los logs completos de cada ejecución están en `my_logs/` (TensorBoard).

## Comparación de arquitecturas

| Arquitectura | Observación |
|---|---|
| ResNet50 + TabNet | **Mejor desempeño general.** Equilibrio entre capacidad de la base y la atención tabular. |
| ResNet50V2 + TabNet | Similar al ganador; la variante V2 no aportó mejora significativa. |
| Xception + TabNet | Buena capacidad de visión, pero más pesada; no superó a ResNet50. |
| Xception + MLP | La rama MLP simple limitó el aprovechamiento de las features tabulares. |
| CNN básica + MLP | Baseline; sobreajuste marcado (F1 train 1.0 vs val 0.56) → motivó el uso de pesos preentrenados. |

## Análisis de equidad (por Fitzpatrick)

El análisis de error por nivel de la escala Fitzpatrick permite cuantificar la tasa de error del modelo a lo largo de la escala de piel (III–VI). Este análisis es central para la motivación del proyecto: verificar que el sistema no degrada su desempeño en los tipos de piel más oscuros.

## Interpretabilidad

- **Grad-CAM:** heatmaps sobre la capa `conv5_block3_out` de ResNet50 que muestran las regiones de la imagen que más contribuyen a la predicción de cada clase.
- **Máscaras de TabNet:** importancia agregada y normalizada de cada feature clínica según las máscaras de atención del encoder, permitiendo identificar qué datos clínicos pesan más en la decisión.

## Figuras

Las figuras de resultados (matriz de confusión, arquitecturas de los modelos) se encuentran en [`../img_in_exp/`](../img_in_exp/).
