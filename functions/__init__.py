"""
Paquete de utilidades reutilizables del proyecto de detección
multimodal de cáncer de piel.

Módulos:
    - config_dataframe: transformador ``BestBinarySplitter`` y utilidades
      para variables categóricas.
    - deep_learning_add: bloques Squeeze-and-Excitation (SE).
    - xception_model: arquitectura Xception construida desde cero.
    - multimodal_config: generador de datos multimodal (imágenes + tabular).
    - multimodal_xception: modelo multimodal Xception completo.
    - utilidades: métricas multiclase y gestión de logs de TensorBoard.
"""

# Los símbolos se exponen de forma *perezosa* (PEP 562) para que importar
# las utilidades puras (p. ej. ``functions.config_dataframe``) NO requiera
# TensorFlow. Los módulos que dependen de TensorFlow solo se cargan cuando
# se accede a su símbolo.
__all__ = [
    "BestBinarySplitter",
    "se_block",
    "multimodal_generator",
    "build_multimodal_xception",
    "get_run_logdir",
    "roc_curve_multiclass",
    "entry_flow",
    "middle_flow",
    "exit_flow",
]

# Mapa símbolo -> (módulo, nombre).
_LAZY_IMPORTS = {
    "BestBinarySplitter": ("functions.config_dataframe", "BestBinarySplitter"),
    "get_run_logdir": ("functions.utilidades", "get_run_logdir"),
    "roc_curve_multiclass": ("functions.utilidades", "roc_curve_multiclass"),
    # --- Símbolos que requieren TensorFlow ---
    "se_block": ("functions.deep_learning_add", "se_block"),
    "entry_flow": ("functions.xception_model", "entry_flow"),
    "middle_flow": ("functions.xception_model", "middle_flow"),
    "exit_flow": ("functions.xception_model", "exit_flow"),
    "multimodal_generator": ("functions.multimodal_config", "multimodal_generator"),
    "build_multimodal_xception": ("functions.multimodal_xception", "build_multimodal_xception"),
}


def __getattr__(name: str):
    """Importa perezosamente el símbolo solicitado (PEP 562)."""
    if name in _LAZY_IMPORTS:
        import importlib

        module_name, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    raise AttributeError(f"módulo {__name__!r} no tiene el atributo {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
