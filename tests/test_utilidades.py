"""
Tests unitarios para ``functions.utilidades``.

Cubre ``get_run_logdir`` (gestión de directorios de TensorBoard) y
``roc_curve_multiclass`` (curvas ROC multiclase con validación cruzada).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend no interactivo para tests

import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from functions.utilidades import get_run_logdir, roc_curve_multiclass


# ---------------------------------------------------------------------------
# get_run_logdir
# ---------------------------------------------------------------------------
def test_get_run_logdir_creates_directory(tmp_path):
    """Debe crear y devolver un directorio de ejecución único."""
    run_dir = Path(get_run_logdir(tmp_path))
    assert run_dir.is_dir()
    assert run_dir.parent == tmp_path
    # El nombre debe seguir el patrón run_YYYY_...
    assert run_dir.name.startswith("run_")


def test_get_run_logdir_creates_parent(tmp_path):
    """Debe crear también los directorios padres que no existan."""
    nested = tmp_path / "a" / "b" / "logs"
    run_dir = Path(get_run_logdir(nested))
    assert run_dir.is_dir()
    assert run_dir.parent == nested


def test_get_run_logdir_name_format(tmp_path):
    """El nombre del directorio debe seguir el patrón run_YYYY_MM_DD-HH_MM_SS."""
    import re

    run_dir = Path(get_run_logdir(tmp_path))
    assert re.fullmatch(r"run_\d{4}_\d{2}_\d{2}-\d{2}_\d{2}_\d{2}", run_dir.name)


# ---------------------------------------------------------------------------
# roc_curve_multiclass
# ---------------------------------------------------------------------------
@pytest.fixture
def multiclass_data():
    """Datos multiclase (3 clases) linealmente separables."""
    rng = np.random.default_rng(42)
    n = 90
    X = np.vstack(
        [
            rng.normal(loc=0, scale=1, size=(n, 2)),
            rng.normal(loc=3, scale=1, size=(n, 2)),
            rng.normal(loc=-3, scale=1, size=(n, 2)),
        ]
    )
    y = np.array([0] * n + [1] * n + [2] * n)
    return X, y


def test_roc_curve_multiclass_runs(multiclass_data):
    """Debe ejecutar sin errores y cerrar la figura."""
    X, y = multiclass_data
    model = LogisticRegression(max_iter=1000)
    roc_curve_multiclass(model, X, y, cv=3)
    # Asegura que no queden figuras abiertas (limpieza).
    plt.close("all")


def test_roc_curve_multiclass_requires_proba():
    """Un clasificador sin predict_proba debe fallar en cross_val_predict."""
    from sklearn.svm import LinearSVC

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    y = np.array([0, 1] * 15)
    # LinearSVC no implementa predict_proba (solo decision_function).
    with pytest.raises(AttributeError):
        roc_curve_multiclass(LinearSVC(max_iter=1000), X, y, cv=3)
    plt.close("all")
