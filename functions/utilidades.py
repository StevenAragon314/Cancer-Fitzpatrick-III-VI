"""
Utilidades generales del proyecto: métricas de clasificación
multiclase y gestión de directorios de logs de TensorBoard.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import cross_val_predict


def roc_curve_multiclass(
    mod: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int | None = None,
    figsize: tuple[float, float] = (10, 8),
) -> None:
    """
    Dibuja las curvas ROC de un clasificador multiclase usando
    validación cruzada (one-vs-rest).

    Args:
        mod: Clasificador scikit-learn con método ``predict_proba``.
        X_train: Matriz de características de entrenamiento.
        y_train: Vector de etiquetas (enteros o one-hot).
        cv: Estrategia de validación cruzada (int, KFold, etc.).
        figsize: Tamaño de la figura resultante.
    """
    # Tomar parámetros únicos
    classes = np.unique(y_train)
    n_classes = len(classes)

    # One-hot de las etiquetas reales (necesario para roc_curve)
    y_train_bin = np.eye(n_classes)[y_train]

    # Obtener probabilidades con validación cruzada
    y_probas = cross_val_predict(
        mod,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )

    # Calcular fpr, tpr, auc para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_train_bin[:, i], y_probas[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_train_bin.ravel(), y_probas.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize= figsize)

    # Macro y Micro primero
    plt.plot(fpr["micro"], tpr["micro"],
            label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
            color='navy', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'Clase: {classes[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC para Clasificación Multiclase')
    plt.legend(loc="lower right")
    plt.grid(False)


def get_run_logdir(root_logdir: str | os.PathLike) -> str:
    """
    Genera un directorio de log único para una ejecución de TensorBoard.

    El nombre incluye la fecha y hora de la ejecución para evitar
    colisiones entre runs.

    Args:
        root_logdir: Directorio raíz donde se crearán los subdirectorios
            de ejecución (por defecto ``my_logs`` en la raíz del proyecto).

    Returns:
        Ruta absoluta del directorio de log recién creado.
    """
    root = Path(root_logdir)
    root.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)