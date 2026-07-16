import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Leemos una imagen y comprueba que sea válida.
def cargar_imagen(ruta: Path) -> np.ndarray:
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")

    imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if imagen is None or imagen.size == 0:
        raise ValueError(f"El archivo no contiene una imagen válida: {ruta}")
    return imagen


# Recortamos solo el borde inferior y derecho.
def ajustar_imagen(imagen: np.ndarray, filas: int, columnas: int) -> np.ndarray:
    if filas < 2 or columnas < 2:
        raise ValueError("Las filas y columnas deben ser como mínimo 2")

    alto, ancho = imagen.shape[:2]
    if alto < filas or ancho < columnas:
        raise ValueError("La imagen es demasiado pequeña para la cuadrícula")

    alto_util = alto - alto % filas
    ancho_util = ancho - ancho % columnas
    return imagen[:alto_util, :ancho_util].copy()


# Dividimos la imagen de izquierda a derecha y de arriba hacia abajo.
def dividir_imagen(imagen: np.ndarray, filas: int, columnas: int) -> list[np.ndarray]:
    alto, ancho = imagen.shape[:2]
    alto_pieza = alto // filas
    ancho_pieza = ancho // columnas
    piezas = []

    for fila in range(filas):
        for columna in range(columnas):
            y = fila * alto_pieza
            x = columna * ancho_pieza
            piezas.append(imagen[y : y + alto_pieza, x : x + ancho_pieza].copy())
    return piezas


# Creamos una mezcla repetible y diferente del orden original.
def mezclar_indices(cantidad: int, semilla: int) -> list[int]:
    original = list(range(cantidad))
    mezcla = original.copy()
    generador = random.Random(semilla)

    while mezcla == original:
        generador.shuffle(mezcla)
    return mezcla


# Colocamos las piezas siguiendo el nuevo orden.
def renderizar(
    piezas: list[np.ndarray], orden: list[int], filas: int, columnas: int
) -> np.ndarray:
    alto_pieza, ancho_pieza = piezas[0].shape[:2]
    lienzo = np.zeros(
        (filas * alto_pieza, columnas * ancho_pieza, 3), dtype=np.uint8
    )

    for posicion, indice in enumerate(orden):
        fila, columna = divmod(posicion, columnas)
        y = fila * alto_pieza
        x = columna * ancho_pieza
        lienzo[y : y + alto_pieza, x : x + ancho_pieza] = piezas[indice]
    return lienzo


# Dibujamos líneas rojas para distinguir las piezas.
def agregar_bordes(imagen: np.ndarray, filas: int, columnas: int) -> np.ndarray:
    resultado = imagen.copy()
    alto, ancho = resultado.shape[:2]

    for fila in range(1, filas):
        y = fila * alto // filas
        cv2.line(resultado, (0, y), (ancho - 1, y), (0, 0, 255), 2)

    for columna in range(1, columnas):
        x = columna * ancho // columnas
        cv2.line(resultado, (x, 0), (x, alto - 1), (0, 0, 255), 2)
    return resultado


# Guardamos una imagen y avisa si OpenCV no puede escribirla.
def guardar_png(ruta: Path, imagen: np.ndarray) -> None:
    if not cv2.imwrite(str(ruta), imagen):
        raise OSError(f"No se pudo guardar: {ruta}")


# Ejecutamos todo el proceso del generador.
def generar_rompecabezas(
    ruta_imagen: Path,
    filas: int,
    columnas: int,
    semilla: int,
    directorio_salida: Path,
) -> dict[str, Any]:
    original = cargar_imagen(ruta_imagen)
    procesada = ajustar_imagen(original, filas, columnas)
    piezas = dividir_imagen(procesada, filas, columnas)
    orden = mezclar_indices(len(piezas), semilla)

    carpeta_piezas = directorio_salida / "pieces"
    carpeta_piezas.mkdir(parents=True, exist_ok=True)

    guardar_png(directorio_salida / "original.png", procesada)
    for indice, pieza in enumerate(piezas):
        guardar_png(carpeta_piezas / f"piece_{indice:03d}.png", pieza)

    rompecabezas = renderizar(piezas, orden, filas, columnas)
    guardar_png(directorio_salida / "shuffled_puzzle.png", rompecabezas)
    guardar_png(
        directorio_salida / "shuffled_puzzle_debug.png",
        agregar_bordes(rompecabezas, filas, columnas),
    )

    alto_original, ancho_original = original.shape[:2]
    alto, ancho = procesada.shape[:2]
    posiciones = {pieza: posicion for posicion, pieza in enumerate(orden)}
    metadata = {
        "imagen_origen": ruta_imagen.name,
        "ancho_original": ancho_original,
        "alto_original": alto_original,
        "ancho_procesado": ancho,
        "alto_procesado": alto,
        "filas": filas,
        "columnas": columnas,
        "cantidad_piezas": len(piezas),
        "ancho_pieza": ancho // columnas,
        "alto_pieza": alto // filas,
        "semilla": semilla,
        "orden_mezclado": orden,
        "piezas": [
            {
                "id": indice,
                "archivo": f"piece_{indice:03d}.png",
                "fila_original": indice // columnas,
                "columna_original": indice % columnas,
                "posicion_mezclada": posiciones[indice],
            }
            for indice in range(len(piezas))
        ],
    }
    (directorio_salida / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {"ancho": ancho, "alto": alto, "cantidad_piezas": len(piezas)}
