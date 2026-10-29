"""
Modelo multimodal Xception.

Combina una rama de visión (Xception construida desde cero en
``functions.xception_model``) con una rama tabular de varias ramas
densas con conexiones residuales, fusionadas por concatenación.

Uso:
    from functions.multimodal_xception import build_multimodal_xception

    model = build_multimodal_xception(img_shape=(256, 256, 3),
                                      tabular_shape=(29,),
                                      num_classes=6)
"""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Input,
    concatenate,
    add,
)
from tensorflow.keras.models import Model

from functions.xception_model import entry_flow, middle_flow, exit_flow


def build_tabular_branch(
    table_input: keras.KerasTensor,
    dropout: float = 0.3,
) -> keras.KerasTensor:
    """
    Construye la rama tabular con tres ramas paralelas y un cabezal
    de bloques residuales estilo Xception.

    Args:
        table_input: Tensor de entrada tabular.
        dropout: Tasa de dropout aplicada en las ramas.

    Returns:
        Tensor de salida de la rama tabular (dimensión 1024).
    """
    # --- RAMA 1 (estrecha) ---
    stem = Dense(32, activation="relu", kernel_initializer="he_normal")(table_input)
    stem = Dense(64, activation="relu", kernel_initializer="he_normal")(stem)

    branch_1 = Dense(128, activation="relu", kernel_initializer="he_normal")(stem)
    branch_1 = Dense(128, activation="relu", kernel_initializer="he_normal")(branch_1)

    # --- RAMA 2 (ancha, con atajo residual) ---
    shortcut = Dense(256, activation="relu", kernel_initializer="he_normal")(stem)
    branch_2 = layers.Dropout(dropout)(shortcut)
    branch_2 = Dense(256, activation="relu", kernel_initializer="he_normal")(branch_2)
    branch_2 = add([shortcut, branch_2])

    # --- RAMA 3 (simple, mira los datos originales) ---
    branch_3 = Dense(128, activation="relu", kernel_initializer="he_normal")(table_input)
    branch_3 = layers.Dropout(dropout + 0.1)(branch_3)

    # --- FUSIÓN DE RAMAS (concatenación) ---
    combined_features = concatenate([branch_1, branch_2, branch_3])

    # --- CABEZAL FINAL ---
    head = Dense(728, activation="relu", kernel_initializer="he_normal")(combined_features)
    head = layers.Dropout(0.5)(head)
    head = Dense(728, activation="relu", kernel_initializer="he_normal")(head)

    # --- BLOQUES RESIDUALES REPETIDOS ---
    head = build_repeated_xception_block(head, repetitions=8)

    # --- SALIDA DE LA RAMA (1024) ---
    out = Dense(728, kernel_initializer="he_normal")(head)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Dense(1024, kernel_initializer="he_normal")(out)
    out = BatchNormalization()(out)
    out = Activation("relu")(out)

    # Conexión residual significativa hacia la salida
    if head.shape[-1] != out.shape[-1]:
        head_resized = Dense(1024)(head)
        out = add([out, head_resized])
    else:
        out = add([out, head])

    return out


def build_repeated_xception_block(input_tensor, repetitions=8):
    """
    Construye un bloque de capas estilo Xception que se repite N veces,
    utilizando conexiones residuales para estabilizar el entrenamiento.

    Args:
        input_tensor: El tensor de entrada al primer bloque.
        repetitions (int): El número de veces que se repetirá el bloque.

    Returns:
        El tensor de salida después de pasar por todos los bloques repetidos.
    """
    # Inicializamos la variable que se actualizará en cada iteración del bucle
    x = input_tensor

    # Bucle que repite el bloque de capas 8 veces
    for _ in range(repetitions):
        # 1. Guarda la entrada del bloque actual para la conexión residual
        residual = x

        x = layers.Dense(728, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = Activation("relu")(x)
        x = layers.Dropout(0.2)(x)

        x = layers.Dense(728, kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = Activation("relu")(x)

        x = layers.add([x, residual])

    return x


def build_multimodal_xception(
    img_shape: tuple[int, int, int] = (256, 256, 3),
    tabular_shape: tuple[int, ...] = (29,),
    num_classes: int = 6,
    head_dropout: float = 0.5,
) -> Model:
    """
    Construye el modelo multimodal completo (Xception + rama tabular).

    Args:
        img_shape: Forma de la imagen de entrada (H, W, C).
        tabular_shape: Forma de la entrada tabular (n_features,).
        num_classes: Número de clases de diagnóstico.
        head_dropout: Tasa de dropout del cabezal final.

    Returns:
        Modelo Keras compilable con dos entradas (imagen, tabular).
    """
    # --- RAMA DE VISIÓN (Xception desde cero) ---
    img_input = Input(shape=img_shape, name="img_xception_input")
    x = entry_flow(img_input)
    x = middle_flow(x)
    vision_branch = exit_flow(x)

    # --- RAMA TABULAR ---
    table_input = Input(shape=tabular_shape, name="table_input_xception")
    tabular_branch = build_tabular_branch(table_input)

    # --- FUSIÓN ---
    combined = concatenate([vision_branch, tabular_branch])

    # --- CABEZAL DE CLASIFICACIÓN ---
    final_head = Dense(512, activation="relu")(combined)
    final_head = layers.Dropout(head_dropout)(final_head)
    final_output = Dense(num_classes, activation="softmax", name="final_output")(final_head)

    model = Model(inputs=[img_input, table_input], outputs=[final_output])
    return model