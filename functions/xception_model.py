"""
Arquitectura Xception construida desde cero (Keras Functional API).

Implementa los tres flujos de la red Xception (entry, middle y exit) tal y
como se describe en el paper *"Xception: Deep Learning with Depthwise
Separable Convolutions"* (Howard et al., 2017). Se usa como rama de visión
del modelo multimodal en :mod:`functions.multimodal_xception`.
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    GlobalAvgPool2D,
    Input,
    MaxPool2D,
    ReLU,
    SeparableConv2D,
)

# Tipo de tensor de la API funcional de Keras.
Tensor = keras.KerasTensor


def conv_bn(x: Tensor, filters: int, kernel_size: int, strides: int = 1) -> Tensor:
    """Bloque Conv2D + BatchNormalization (sin sesgo)."""
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def sep_bn(x: Tensor, filters: int, kernel_size: int, strides: int = 1) -> Tensor:
    """Bloque SeparableConv2D + BatchNormalization (sin sesgo)."""
    x = SeparableConv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def entry_flow(x: Tensor) -> Tensor:
    x = conv_bn(x, filters=32, kernel_size=3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters=64, kernel_size=3)
    tensor = ReLU()(x)
 
    x = sep_bn(tensor, filters=128, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=128, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
 
    tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)
 
    x = Add()([tensor, x])
    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
 
    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)
 
    x = Add()([tensor, x])
    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)
 
    tensor = conv_bn(tensor, filters=728, kernel_size=1, strides=2)
    x = Add()([tensor, x])
 
    return x
 
 
def middle_flow(tensor: Tensor) -> Tensor:
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)
 
        tensor = Add()([tensor, x])
 
    return tensor
 
 
def exit_flow(tensor: Tensor) -> Tensor:
    x = ReLU()(tensor)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=1024, kernel_size=3)
    x = MaxPool2D(3, strides=2, padding='same')(x)
 
    tensor = conv_bn(tensor, filters=1024, kernel_size=1, strides=2)
 
    x = Add()([tensor, x])
    x = sep_bn(x, filters=1536, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=2048, kernel_size=3)
    x = ReLU()(x)
    x = GlobalAvgPool2D()(x)
    
    return x