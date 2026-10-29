"""
Bloques de atención reutilizables para redes convolucionales.

Actualmente incluye el bloque Squeeze-and-Excitation (SE) de Hu et al.
(2018), que re-pondera los canales de un tensor convolucional para
destacar la información más relevante.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Reshape, multiply

# Tipo de tensor de la API funcional de Keras.
Tensor = keras.KerasTensor


def se_block(input_tensor: Tensor, ratio: int = 16) -> Tensor:
    """
    Crea un bloque Squeeze-and-Excitation.

    Args:
        input_tensor: El tensor de entrada (salida de una capa convolucional).
        ratio (int): El factor de reducción para la capa oculta del bloque SE.

    Returns:
        El tensor de salida con los canales re-ponderados.
    """
    # Obtiene el número de canales del tensor de entrada
    channels = input_tensor.shape[-1]

    # --- Squeeze ---
    # Comprime la información espacial en un solo valor por canal
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, channels))(se) # Reajusta la forma para las capas densas

    # --- Excitation ---
    # Reduce la dimensionalidad
    se = Dense(channels // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    # Vuelve a la dimensionalidad original para obtener los pesos
    se = Dense(channels, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    # Aplica los pesos (la "atención") al tensor de entrada original
    output_tensor = multiply([input_tensor, se])
    return output_tensor