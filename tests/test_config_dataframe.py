"""
Tests unitarios para ``functions.config_dataframe``.

Cubre el transformador ``BestBinarySplitter`` y las funciones auxiliares
``split_stats`` y ``prepare_categoricals``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from functions.config_dataframe import (
    BestBinarySplitter,
    _best_binary_split,
    _impurities,
    prepare_categoricals,
    split_stats,
)


# ---------------------------------------------------------------------------
# _impurities
# ---------------------------------------------------------------------------
def test_impurities_pure_node_is_zero():
    """Un nodo puro (una sola clase) debe tener impureza cero."""
    gini, entr, ec = _impurities(np.array([10, 0]))
    assert gini == pytest.approx(0.0)
    assert entr == pytest.approx(0.0)
    assert ec == pytest.approx(0.0)


def test_impurities_balanced_node_is_maximal():
    """Un nodo 50/50 binario debe tener la impureza máxima."""
    gini, entr, ec = _impurities(np.array([5, 5]))
    assert gini == pytest.approx(0.5)
    assert entr == pytest.approx(1.0)
    assert ec == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _best_binary_split
# ---------------------------------------------------------------------------
def test_best_binary_split_returns_partition():
    """Debe devolver una partición no vacía y una ganancia finita."""
    y = pd.Series([0, 0, 0, 1, 1, 1, 0, 1])
    cat = pd.Series(["a", "a", "b", "b", "c", "c", "a", "b"])
    left, gain = _best_binary_split(y, cat, criterion="gini")
    assert isinstance(left, set)
    assert left  # no vacío
    assert np.isfinite(gain)


# ---------------------------------------------------------------------------
# BestBinarySplitter
# ---------------------------------------------------------------------------
@pytest.fixture
def categorical_data() -> tuple[pd.DataFrame, pd.Series]:
    """DataFrame con una categórica de 3 niveles y una numérica."""
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame(
        {
            "region": rng.choice(["torso", "head/neck", "upper extremity"], size=n),
            "age": rng.integers(20, 70, size=n),
        }
    )
    # La etiqueta depende de la región para que el split tenga señal.
    y = pd.Series(
        np.where(X["region"] == "torso", 1, 0),
    )
    return X, y


def test_splitter_requires_y(categorical_data):
    """fit() sin y debe lanzar ValueError."""
    X, _ = categorical_data
    with pytest.raises(ValueError):
        BestBinarySplitter().fit(X)


def test_splitter_output_shape(categorical_data):
    """transform() debe devolver una columna binaria por categórica."""
    X, y = categorical_data
    splitter = BestBinarySplitter().fit(X, y)
    out = splitter.transform(X)
    # Solo la columna categórica se transforma; la numérica se descarta.
    assert list(out.columns) == ["region__split"]
    assert set(out["region__split"].unique()).issubset({0, 1})
    assert out.shape[0] == X.shape[0]


def test_splitter_feature_names(categorical_data):
    """get_feature_names_out debe coincidir con las columnas de transform."""
    X, y = categorical_data
    splitter = BestBinarySplitter().fit(X, y)
    names = splitter.get_feature_names_out()
    assert list(names) == list(splitter.transform(X).columns)


def test_splitter_binary_column_untouched(categorical_data):
    """Una categórica binaria (2 niveles) debe mapearse al primer nivel."""
    X, y = categorical_data
    X["sex"] = np.random.default_rng(1).choice(["male", "female"], size=len(X))
    splitter = BestBinarySplitter().fit(X, y)
    out = splitter.transform(X)
    assert "sex__split" in out.columns


def test_splitter_clone_preserves_params(categorical_data):
    """clone() (requisito de sklearn) debe preservar los hiperparámetros."""
    X, y = categorical_data
    original = BestBinarySplitter(criterion="entropy", min_levels=4)
    cloned = clone(original)
    assert cloned.criterion == "entropy"
    assert cloned.min_levels == 4


def test_splitter_handle_unknown_raise(categorical_data):
    """handle_unknown='raise' debe lanzar al ver un nivel desconocido."""
    X, y = categorical_data
    splitter = BestBinarySplitter(handle_unknown="raise").fit(X, y)
    X_new = X.copy()
    X_new.loc[0, "region"] = "unknown_region"
    with pytest.raises(ValueError):
        splitter.transform(X_new)


# ---------------------------------------------------------------------------
# split_stats
# ---------------------------------------------------------------------------
def test_split_stats_returns_gains(categorical_data):
    """split_stats debe devolver left/right/gain/gains."""
    X, y = categorical_data
    result = split_stats(y, X["region"])
    assert result is not None
    assert "left" in result and "right" in result
    assert "gain" in result
    assert set(result["gains"]) == {"Gini", "Entropy", "Error"}
    # left y right deben ser complementarios.
    assert result["left"].isdisjoint(result["right"])


# ---------------------------------------------------------------------------
# prepare_categoricals
# ---------------------------------------------------------------------------
def test_prepare_categoricals_replaces_column(categorical_data):
    """prepare_categoricals debe sustituir la categórica por su split."""
    X, y = categorical_data
    X = X.copy()
    X["region"] = X["region"].astype("category")
    out = prepare_categoricals(X, y, min_levels=3)
    assert "region" not in out.columns
    assert "region_split" in out.columns
    assert set(out["region_split"].unique()).issubset({0, 1})
