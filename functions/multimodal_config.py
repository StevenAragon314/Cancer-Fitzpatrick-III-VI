"""
Generador de datos multimodal (imágenes + datos tabulares).

Produce lotes sincronizados de imágenes y filas tabulares, listos para
``tf.data.Dataset.from_generator``.
"""

from __future__ import annotations

import os
from typing import Any, Generator

import numpy as np
from numpy import arange, array
from numpy.random import shuffle
from pandas import DataFrame
from warnings import warn

from tensorflow.keras.preprocessing.image import load_img, img_to_array


def multimodal_generator(
    dataframe: DataFrame,
    train_datagen: Any,
    tabular_data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    img_path: str,
    img_size: tuple[int, int] = (224, 224),
    augment: bool = False,
) -> Generator[tuple[tuple[np.ndarray, np.ndarray], np.ndarray], None, None]:
    """
    Crea un generador de datos para un modelo multimodal que procesa imágenes y datos tabulares.

    Esta función produce lotes de datos perfectamente sincronizados, asegurando que cada imagen se corresponda
    con su fila de datos tabulares y su etiqueta. Es ideal para ser utilizada con `tf.data.Dataset.from_generator`.

    Args:
        dataframe (pd.DataFrame): Un DataFrame de pandas que contiene los identificadores de las imágenes
            y, potencialmente, otras metadata. Debe tener una columna con los nombres de los archivos de imagen.

        train_datagen (ImageDataGenerator): Este objeto se desarrolla con ``tensorflow.keras.preprocessing.imag``,
            en este se tiene que construir con la información escencial para capturar las información de las imagenes
            a analizar.

        tabular_data (np.ndarray): Un array de NumPy con los datos tabulares ya preprocesados (escalados,
            codificados, etc.). El orden de las filas DEBE corresponder exactamente con el orden del `dataframe`.

        labels (np.ndarray): Un array de NumPy que contiene las etiquetas de los datos, ya en formato
            numérico y listas para el modelo (ej. one-hot encoded). El orden también DEBE corresponder
            con el del `dataframe`.

        batch_size (int): El número de muestras a incluir en cada lote que produce el generador.

        img_path (str): La ruta a la carpeta (directorio) donde están almacenadas todas las imágenes.

        img_size (tuple, optional): Una tupla de dos enteros `(altura, anchura)` a la cual se redimensionarán
            todas las imágenes. Por defecto es `(224, 224)`.

        augment (bool, optional): Si es `True`, se aplicará aumentación de datos a las imágenes del lote
            y se barajarán los datos en cada época. Si es `False`, solo se cargarán y normalizarán las
            imágenes sin alterarlas. Por defecto es `False`. Se debe usar `True` para el generador de
            entrenamiento y `False` para el de validación/prueba.
            
    Yields:
        tuple: Una tupla con la estructura `((batch_images, batch_tabular), batch_labels)`.
            - El primer elemento es otra tupla que contiene las dos entradas del modelo:
                - batch_images (np.ndarray): Un lote de imágenes con forma (batch_size, altura, anchura, 3).
                - batch_tabular (np.ndarray): Un lote de datos tabulares con forma (batch_size, num_features_tabulares).
            - El segundo elemento contiene las etiquetas correspondientes:
                - batch_labels (np.ndarray): Un lote de etiquetas con forma (batch_size, num_clases).
    """

    if not os.path.isdir(img_path):
        raise FileNotFoundError(f"El directorio de imágenes especificado no existe: '{img_path}'")

    # Valida que todos los datos de entrada tengan la misma cantidad de muestras
    if not (len(dataframe) == len(tabular_data) == len(labels)):
        raise ValueError(
            "Inconsistencia en los datos de entrada. El DataFrame, los datos tabulares y las etiquetas "
            f"deben tener la misma longitud. Longitudes actuales: DataFrame={len(dataframe)}, "
            f"Tabular={len(tabular_data)}, Labels={len(labels)}"
        )
    
    # Valida que haya suficientes datos para al menos un lote
    if len(dataframe) == 0:
        # Usamos yield para eficiencia en la memoria
        warn("El DataFrame de entrada está vacío. El generador no producirá datos.")
        return

    # --- LÓGICA DEL GENERADOR ---
    num_samples = len(dataframe)
    indices = arange(num_samples)

    while True:
        if augment:
            shuffle(indices)

        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset + batch_size]
            
            # Si el último lote es más pequeño, no hay problema, pero nos aseguramos de no procesar un lote vacío
            if len(batch_indices) == 0:
                continue

            batch_tabular = tabular_data[batch_indices]
            batch_labels = labels[batch_indices]
            
            batch_images = []
            valid_indices_in_batch = [] # Guardaremos los índices que sí funcionaron

            filenames = dataframe['img_id'].iloc[batch_indices]

            for i, filename in enumerate(filenames):
                img_full_path = os.path.join(img_path, str(filename)) # Aseguramos que el nombre sea string
                
                # --- MANEJO DE ERRORES POR ARCHIVO ---
                try:
                    # Carga y procesa la imagen
                    img = load_img(img_full_path, target_size=img_size)
                    img_array = img_to_array(img)
                    img_array = img_array / 255.0
                    
                    batch_images.append(img_array)
                    # Si la imagen se cargó bien, guardamos el índice original del lote
                    valid_indices_in_batch.append(i)

                except FileNotFoundError:
                    # Si no se encuentra un archivo, imprime una advertencia y lo omite
                    warn(f"¡Archivo no encontrado! Se omitirá: '{img_full_path}'")
                    continue # Pasa al siguiente archivo en el lote
                except Exception as e:
                    # Captura otros posibles errores (ej. imagen corrupta)
                    warn(f"No se pudo procesar el archivo '{img_full_path}'. Error: {e}. Se omitirá.")
                    continue

            # Si después de los filtros no quedó ninguna imagen válida en el lote, pasamos al siguiente
            if not batch_images:
                continue

            # Convertimos la lista de imágenes a un array
            batch_images = array(batch_images)

            # Sincronizamos los datos tabulares y las etiquetas para que coincidan con las imágenes que SÍ se cargaron
            batch_tabular = batch_tabular[valid_indices_in_batch]
            batch_labels = batch_labels[valid_indices_in_batch]

            if augment and len(batch_images) > 0:
                # El batch_size para el flow debe ser el número de imágenes válidas
                current_batch_size = len(batch_images)
                batch_images = next(train_datagen.flow(batch_images, shuffle=False, batch_size=current_batch_size))
            
            yield ((batch_images, batch_tabular), batch_labels)