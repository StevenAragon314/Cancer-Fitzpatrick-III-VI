"""
Transformadores y utilidades para el tratamiento de variables categóricas.

El componente central es :class:`BestBinarySplitter`, un transformer de
scikit-learn que convierte variables categóricas en variables binarias 0/1
buscando la partición que maximiza la ganancia de impureza (Gini, entropía
o error de clasificación). Es útil para modelos lineales y LDA, donde el
one-hot de variables de alta cardinalidad dispara la dimensionalidad.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# -- utilidades internas ---------------------------------------------------
def _impurities(counts: np.ndarray) -> tuple[float, float, float]:
    """Calcula Gini, entropía y error de clasificación para un recuento de clases."""
    n = counts.sum()
    p = counts / n
    gini = 1 - (p**2).sum()
    entr = -(p * np.where(p == 0, 1, np.log2(p))).sum()
    ec   = 1 - p.max()
    return gini, entr, ec


def _best_binary_split(
    y: pd.Series,
    cat: pd.Series,
    criterion: str = "gini",
) -> tuple[set, float]:
    tab_full = pd.crosstab(y, cat)
    k = tab_full.shape[1]
    parent_imp = _impurities(tab_full.sum(axis=1).values)

    best_gain, best_left = -np.inf, None
    for r in range(1, k // 2 + 1):
        for left in combinations(tab_full.columns, r):
            mask = cat.isin(left)
            n_l, n_r = mask.sum(), (~mask).sum()
            if n_l == 0 or n_r == 0:
                continue

            imp_l = _impurities(pd.crosstab(y[mask], cat[mask])
                                .sum(axis=1).values)
            imp_r = _impurities(pd.crosstab(y[~mask], cat[~mask])
                                .sum(axis=1).values)

            if criterion == "gini":
                split_imp = (n_l * imp_l[0] + n_r * imp_r[0]) / (n_l + n_r)
                gain = parent_imp[0] - split_imp
            elif criterion == "entropy":
                split_imp = (n_l * imp_l[1] + n_r * imp_r[1]) / (n_l + n_r)
                gain = parent_imp[1] - split_imp
            else:  # error clas.
                split_imp = (n_l * imp_l[2] + n_r * imp_r[2]) / (n_l + n_r)
                gain = parent_imp[2] - split_imp

            if gain > best_gain:
                best_gain, best_left = gain, set(left)

    return best_left, best_gain


# -- transformer ----------------------------------------------------------
class BestBinarySplitter(BaseEstimator, TransformerMixin):
    """
    Convierte variables categóricas en 0/1 buscando la partición binaria que
    maximiza la ganancia de impureza (Gini, entropía o error de clasificación).
    """
    def __init__(
        self,
        criterion: str = "gini",
        min_levels: int = 3,
        handle_unknown: str = "zero",
    ) -> None:
        self.criterion = criterion
        self.min_levels = min_levels
        self.handle_unknown = handle_unknown  # "zero" o "raise"

    # ------------ ajuste ---------------------------------------------------
    def fit(self, X: Any, y: Any = None) -> "BestBinarySplitter":
        if y is None:
            raise ValueError("Necesito `y` para evaluar la ganancia de impureza.")
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        self.splits_ = {}
        self.binary_cols_ = []
        self.known_levels_ = {}

        for col in X.columns:
            if (
                X[col].dtype.kind not in "O"
                and not pd.api.types.is_categorical_dtype(X[col])
            ):
                continue  # no categórica

            k = X[col].nunique(dropna=False)
            if k < 2:
                continue
            if k == 2:
                left = {X[col].unique()[0]}
            elif k >= self.min_levels:
                left, _ = _best_binary_split(y, X[col], self.criterion)
            else:                       # 3 ≤ k < min_levels
                left = {X[col].unique()[0]}

            self.splits_[col] = left
            self.known_levels_[col] = set(X[col].unique())
            self.binary_cols_.append(f"{col}__split")

        return self

    # ------------ transformación ------------------------------------------
    def transform(self, X: Any) -> pd.DataFrame:
        X = pd.DataFrame(X).reset_index(drop=True)
        out = pd.DataFrame(index=X.index)

        for col, left in self.splits_.items():
            if self.handle_unknown == "raise":
                # Niveles que aparecen en X pero no se vieron en fit().
                unseen = set(X[col].unique()) - self.known_levels_[col]
                if unseen:
                    raise ValueError(f"Niveles desconocidos en columna {col}: {unseen}")
            out[f"{col}__split"] = X[col].isin(left).astype(int)

        return out

    # ------------ nombres de salida ---------------------
    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(self.binary_cols_)


def split_stats(y: pd.Series, cat: pd.Series) -> dict[str, Any] | None:
    """
    Devuelve la mejor partición binaria de un factor 'cat' respecto a 'y'
    y las ganancias de impureza correspondientes.
    """
    y = pd.Series(y)               # asegura alineación
    cat = pd.Series(cat)

    # matriz de contingencia completa
    full_tab = pd.crosstab(y, cat)
    g_p, e_p, ec_p = _impurities(full_tab.sum(axis=1).values)

    best = None
    levels = full_tab.columns
    k = len(levels)

    # explora TODAS las particiones no triviales: 2^(k-1) - 1
    for r in range(1, k // 2 + 1):
        for left in combinations(levels, r):
            mask = cat.isin(left)
            tab_l = pd.crosstab(y[mask], cat[mask])
            tab_r = pd.crosstab(y[~mask], cat[~mask])

            # tab_l / tab_r podrían perder niveles ⇒ reindex
            tab_l = tab_l.reindex(index=full_tab.index, fill_value=0)
            tab_r = tab_r.reindex(index=full_tab.index, fill_value=0)

            n_l, n_r = mask.sum(), (~mask).sum()
            n = n_l + n_r

            g_l, e_l, ec_l = _impurities(tab_l.sum(axis=1).values)
            g_r, e_r, ec_r = _impurities(tab_r.sum(axis=1).values)

            g_split = (n_l / n) * g_l + (n_r / n) * g_r
            e_split = (n_l / n) * e_l + (n_r / n) * e_r
            ec_split = (n_l / n) * ec_l + (n_r / n) * ec_r

            gain_g = g_p - g_split
            gain_e = e_p - e_split
            gain_ec = ec_p - ec_split

            score = gain_g          # usa la que prefieras
            if best is None or score > best["gain"]:
                best = dict(
                    left=set(left),
                    right=set(levels) - set(left),
                    gain=score,
                    gains=dict(Gini=gain_g, Entropy=gain_e, Error= gain_ec)
                )
    return best

def prepare_categoricals(df: pd.DataFrame, y: pd.Series, min_levels: int = 3) -> pd.DataFrame:
    """
    Reemplaza cada variable categórica por su mejor split binario.
    Devuelve un nuevo DataFrame apto para modelos lineales, LDA, etc.
    """
    df_new = df.copy()
    for col in df.select_dtypes(include="category"):
        if df[col].nunique() < min_levels:
            continue                        # ya es binaria/ternaria

        split = split_stats(y, df[col])
        if split is None:
            continue

        # crea la variable binaria 0/1
        name = f"{col}_split"
        df_new[name] = df[col].isin(split["left"]).astype(int)
        df_new = df_new.drop(columns=col)

        print(f"{col}: {split['left']} vs {split['right']} — "
              f"Gini = {split['gains']['Gini']:.4f}")
    return df_new

def explica_fila(idx: int, X_df: Any, y_ser: pd.Series, pipe: Any) -> None:
    """
    Imprime la ruta que sigue la observación `idx` en el árbol que vive
    dentro de `pipe`.  Soporta `X_df` como DataFrame o como ndarray.

    Parameters
    ----------
    idx   : int
        Índice (posición) de la observación que quieres explicar.
    X_df  : pandas.DataFrame | np.ndarray
        Conjunto de predictores sobre el que se entrenó el pipeline.
    y_ser : pandas.Series
        Series con la respuesta real.
    pipe  : sklearn.pipeline.Pipeline
        Pipeline cuyo último paso es DecisionTreeClassifier y que contiene
        un ColumnTransformer en el paso 'columntransformer'.
    """
    # -- 1.  Asegurar que X_df sea DataFrame -----------------------------
    if isinstance(X_df, np.ndarray):
        # intentamos recuperar los nombres originales del preprocesador
        pre = pipe.named_steps["columntransformer"]
        if hasattr(pre, "feature_names_in_"):
            cols = pre.feature_names_in_
        else:                               # fallback genérico
            cols = [f"x{i}" for i in range(X_df.shape[1])]
        X_df = pd.DataFrame(X_df, columns=cols)

    # -- 2.  Objetos internos necesarios -----------------------------------
    pre  = pipe.named_steps["columntransformer"]
    tree = pipe.named_steps["decisiontreeclassifier"]
    feats = pre.get_feature_names_out()

    # -- 3.  Transformar la fila y obtener la ruta -------------------------
    x_row   = X_df.iloc[[idx]]
    x_trans = pre.transform(x_row)
    nodes   = tree.decision_path(x_trans).indices

    # -- 4.  Imprimir paso a paso ------------------------------------------
    print(f"\nFila {idx}  (real = {y_ser.iloc[idx]})")
    for nid in nodes:
        if tree.tree_.children_left[nid] == -1:                 # hoja
            probs = tree.tree_.value[nid][0]
            probs = (probs / probs.sum()).round(3)
            pred  = tree.classes_[probs.argmax()]
            print(f"  → Hoja {nid}  pred = {pred}  p = {probs}")
        else:                                                   # nodo interno
            fid  = tree.tree_.feature[nid]
            thr  = tree.tree_.threshold[nid]
            val  = x_trans[0, fid]
            op   = "≤" if val <= thr else ">"
            side = "izq" if val <= thr else "der"
            print(
                f"  Nodo {nid}: {feats[fid]} {op} {thr:.3f} "
                f"(val={val:.3f}) → {side}"
            )