import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Pieza:
    id: int
    imagen: np.ndarray
    fila: int
    columna: int
    grupo: int
    vertices: list[list[int]]
    x: int = 0
    y: int = 0
    vertices_globales: list[list[int]] | None = None


# Leemos una imagen y comprueba que sea valida.
def cargar_imagen(ruta: Path) -> np.ndarray:
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la imagen: {ruta}")

    imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if imagen is None or imagen.size == 0:
        raise ValueError(f"El archivo no contiene una imagen valida: {ruta}")
    return imagen


# Recortamos solo el borde inferior y derecho.
def ajustar_imagen(imagen: np.ndarray, filas: int, columnas: int) -> np.ndarray:
    if filas < 2 or columnas < 2:
        raise ValueError("Las filas y columnas deben ser como minimo 2")

    alto, ancho = imagen.shape[:2]
    if alto < filas or ancho < columnas:
        raise ValueError("La imagen es demasiado pequena para la cuadricula")

    alto_util = alto - alto % filas
    ancho_util = ancho - ancho % columnas
    return imagen[:alto_util, :ancho_util].copy()


# Define las figuras que ocupan completamente cada celda.
def obtener_figuras(ancho: int, alto: int, lados: int) -> list[list[list[int]]]:
    derecha = ancho - 1
    abajo = alto - 1

    if lados == 3:
        return [
            [[0, 0], [derecha, 0], [0, abajo]],
            [[derecha, 0], [derecha, abajo], [0, abajo]],
        ]

    if lados == 4:
        return [[[0, 0], [derecha, 0], [derecha, abajo], [0, abajo]]]

    if lados == 6:
        centro = derecha // 2
        margen = max(1, derecha // 6)
        tercio = abajo // 3
        dos_tercios = 2 * abajo // 3
        return [
            [
                [0, 0],
                [centro, 0],
                [centro + margen, tercio],
                [centro - margen, dos_tercios],
                [centro, abajo],
                [0, abajo],
            ],
            [
                [centro, 0],
                [derecha, 0],
                [derecha, abajo],
                [centro, abajo],
                [centro - margen, dos_tercios],
                [centro + margen, tercio],
            ],
        ]

    raise ValueError("Solo se permiten figuras de 3, 4 o 6 lados")


# Aplicamos transparencia fuera de los vertices de la pieza.
def recortar_figura(imagen: np.ndarray, vertices: list[list[int]]) -> np.ndarray:
    mascara = np.zeros(imagen.shape[:2], dtype=np.uint8)
    puntos = np.array(vertices, dtype=np.int32)
    cv2.fillPoly(mascara, [puntos], 255)
    pieza = cv2.cvtColor(imagen, cv2.COLOR_BGR2BGRA)
    pieza[:, :, 3] = mascara
    return pieza


# Dividimos cada celda en una o dos figuras segun sus lados.
def dividir_imagen(
    imagen: np.ndarray, filas: int, columnas: int, lados: int
) -> list[Pieza]:
    alto, ancho = imagen.shape[:2]
    alto_celda = alto // filas
    ancho_celda = ancho // columnas
    figuras = obtener_figuras(ancho_celda, alto_celda, lados)
    piezas = []

    for fila in range(filas):
        for columna in range(columnas):
            y = fila * alto_celda
            x = columna * ancho_celda
            celda = imagen[y : y + alto_celda, x : x + ancho_celda]

            for grupo, vertices in enumerate(figuras):
                piezas.append(
                    Pieza(
                        id=len(piezas),
                        imagen=recortar_figura(celda, vertices),
                        fila=fila,
                        columna=columna,
                        grupo=grupo,
                        vertices=vertices,
                        x=x,
                        y=y,
                        vertices_globales=[[vx + x, vy + y] for vx, vy in vertices],
                    )
                )
    return piezas


def _generar_puntos_voronoi(
    ancho: int, alto: int, cantidad_piezas: int, semilla: int
) -> list[tuple[int, int]]:
    if cantidad_piezas < 2:
        raise ValueError("Voronoi necesita al menos 2 piezas")
    if ancho < 4 or alto < 4:
        raise ValueError("La imagen es demasiado pequena para Voronoi")

    margen_x = max(1, min(ancho // 20, ancho // 2 - 1))
    margen_y = max(1, min(alto // 20, alto // 2 - 1))
    generador = random.Random(semilla)
    puntos: list[tuple[int, int]] = []
    intentos = 0
    distancia_minima = max(4, int(min(ancho, alto) / (cantidad_piezas**0.5 * 3)))

    while len(puntos) < cantidad_piezas and intentos < cantidad_piezas * 80:
        x = generador.randint(margen_x, ancho - margen_x - 1)
        y = generador.randint(margen_y, alto - margen_y - 1)
        if all((x - px) ** 2 + (y - py) ** 2 >= distancia_minima**2 for px, py in puntos):
            puntos.append((x, y))
        intentos += 1

    while len(puntos) < cantidad_piezas:
        puntos.append((generador.randint(1, ancho - 2), generador.randint(1, alto - 2)))
    return puntos


def _ordenar_vertices(vertices: list[list[int]]) -> list[list[int]]:
    centro_x = sum(p[0] for p in vertices) / len(vertices)
    centro_y = sum(p[1] for p in vertices) / len(vertices)
    return sorted(vertices, key=lambda p: np.arctan2(p[1] - centro_y, p[0] - centro_x))


# Dividimos la imagen usando diagramas de Voronoi.
def dividir_voronoi(imagen: np.ndarray, cantidad_piezas: int, semilla: int) -> list[Pieza]:
    alto, ancho = imagen.shape[:2]
    rect = (0, 0, ancho, alto)
    subdiv = cv2.Subdiv2D(rect)

    for x, y in _generar_puntos_voronoi(ancho, alto, cantidad_piezas, semilla):
        subdiv.insert((x, y))

    facetas, _ = subdiv.getVoronoiFacetList([])
    piezas = []

    for faceta in facetas:
        if len(faceta) == 0:
            continue

        puntos: list[list[int]] = []
        for punto in faceta:
            px = max(0, min(ancho - 1, int(punto[0])))
            py = max(0, min(alto - 1, int(punto[1])))
            puntos.append([px, py])
        puntos = _ordenar_vertices(puntos)

        x_coords = [p[0] for p in puntos]
        y_coords = [p[1] for p in puntos]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        if x_max <= x_min or y_max <= y_min:
            continue

        vertices_relativos = [[px - x_min, py - y_min] for px, py in puntos]
        recorte = imagen[y_min : y_max + 1, x_min : x_max + 1]
        pieza_img = recortar_figura(recorte, vertices_relativos)

        piezas.append(
            Pieza(
                id=len(piezas),
                imagen=pieza_img,
                fila=y_min,
                columna=x_min,
                grupo=0,
                vertices=vertices_relativos,
                x=x_min,
                y=y_min,
                vertices_globales=puntos,
            )
        )
    return piezas


# Mezclamos por separado las piezas que tienen la misma orientacion.
def mezclar_piezas(piezas: list[Pieza], semilla: int) -> list[int]:
    orden = list(range(len(piezas)))
    generador = random.Random(semilla)
    grupos = sorted({pieza.grupo for pieza in piezas})

    for grupo in grupos:
        posiciones = [pieza.id for pieza in piezas if pieza.grupo == grupo]
        if len(posiciones) < 2:
            continue
        mezcla = posiciones.copy()
        while mezcla == posiciones:
            generador.shuffle(mezcla)
        for posicion, pieza_id in zip(posiciones, mezcla):
            orden[posicion] = pieza_id
    return orden


# Copiamos una pieza transparente sobre el lienzo.
def colocar_pieza(lienzo: np.ndarray, pieza: np.ndarray, x: int, y: int) -> None:
    alto, ancho = pieza.shape[:2]
    lienzo_alto, lienzo_ancho = lienzo.shape[:2]
    x_fin = min(x + ancho, lienzo_ancho)
    y_fin = min(y + alto, lienzo_alto)
    if x >= lienzo_ancho or y >= lienzo_alto or x_fin <= x or y_fin <= y:
        return

    pieza_visible = pieza[: y_fin - y, : x_fin - x]
    mascara = pieza_visible[:, :, 3] > 0
    zona = lienzo[y:y_fin, x:x_fin]
    zona[mascara] = pieza_visible[:, :, :3][mascara]


# Colocamos las piezas en las posiciones mezcladas para piezas de grilla.
def renderizar(
    piezas: list[Pieza], orden: list[int], filas: int, columnas: int
) -> np.ndarray:
    alto_celda, ancho_celda = piezas[0].imagen.shape[:2]
    lienzo = np.zeros(
        (filas * alto_celda, columnas * ancho_celda, 3), dtype=np.uint8
    )

    for posicion, pieza_id in enumerate(orden):
        destino = piezas[posicion]
        pieza = piezas[pieza_id]
        y = destino.fila * alto_celda
        x = destino.columna * ancho_celda
        colocar_pieza(lienzo, pieza.imagen, x, y)
    return lienzo


# Colocamos piezas irregulares en los origenes de otras piezas Voronoi.
def renderizar_en_origen(
    piezas: list[Pieza], orden: list[int], ancho: int, alto: int
) -> np.ndarray:
    lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)
    for destino_id, pieza_id in enumerate(orden):
        destino = piezas[destino_id]
        pieza = piezas[pieza_id]
        colocar_pieza(lienzo, pieza.imagen, destino.x, destino.y)
    return lienzo


# Dibujamos el contorno de cada figura.
def agregar_bordes(imagen: np.ndarray, piezas: list[Pieza]) -> np.ndarray:
    resultado = imagen.copy()

    for pieza in piezas:
        vertices = pieza.vertices_globales or [[x + pieza.x, y + pieza.y] for x, y in pieza.vertices]
        puntos = np.array(vertices, dtype=np.int32)
        cv2.polylines(resultado, [puntos], True, (0, 0, 255), 2)
    return resultado


# Guarda una imagen y avisa si OpenCV no puede escribirla.
def guardar_png(ruta: Path, imagen: np.ndarray) -> None:
    if not cv2.imwrite(str(ruta), imagen):
        raise OSError(f"No se pudo guardar: {ruta}")


# Ejecutamos todo el proceso del generador.
def generar_rompecabezas(
    ruta_imagen: Path,
    filas: int,
    columnas: int,
    lados: int,
    semilla: int,
    directorio_salida: Path,
) -> dict[str, Any]:
    original = cargar_imagen(ruta_imagen)
    procesada = ajustar_imagen(original, filas, columnas)
    if lados == 0:
        piezas = dividir_voronoi(procesada, filas * columnas, semilla)
    else:
        piezas = dividir_imagen(procesada, filas, columnas, lados)
    orden = mezclar_piezas(piezas, semilla)

    carpeta_piezas = directorio_salida / "pieces"
    carpeta_piezas.mkdir(parents=True, exist_ok=True)
    for archivo_anterior in carpeta_piezas.glob("piece_*.png"):
        archivo_anterior.unlink()

    guardar_png(directorio_salida / "original.png", procesada)
    for pieza in piezas:
        guardar_png(carpeta_piezas / f"piece_{pieza.id:03d}.png", pieza.imagen)

    alto_original, ancho_original = original.shape[:2]
    alto, ancho = procesada.shape[:2]
    if lados == 0:
        rompecabezas = renderizar_en_origen(piezas, orden, ancho, alto)
    else:
        rompecabezas = renderizar(piezas, orden, filas, columnas)
    guardar_png(directorio_salida / "shuffled_puzzle.png", rompecabezas)
    guardar_png(
        directorio_salida / "shuffled_puzzle_debug.png",
        agregar_bordes(rompecabezas, piezas),
    )

    posiciones = {pieza_id: posicion for posicion, pieza_id in enumerate(orden)}
    nombres = {0: "voronoi", 3: "triangulo", 4: "cuadrilatero", 6: "hexagono"}
    metadata = {
        "imagen_origen": ruta_imagen.name,
        "ancho_original": ancho_original,
        "alto_original": alto_original,
        "ancho_procesado": ancho,
        "alto_procesado": alto,
        "filas": filas if lados != 0 else None,
        "columnas": columnas if lados != 0 else None,
        "lados": lados,
        "tipo_pieza": nombres[lados],
        "cantidad_piezas": len(piezas),
        "semilla": semilla,
        "orden_mezclado": orden,
        "piezas": [
            {
                "id": pieza.id,
                "archivo": f"piece_{pieza.id:03d}.png",
                "fila_original": pieza.fila if lados != 0 else None,
                "columna_original": pieza.columna if lados != 0 else None,
                "x_origen": pieza.x,
                "y_origen": pieza.y,
                "grupo_orientacion": pieza.grupo,
                "vertices": pieza.vertices,
                "vertices_globales": pieza.vertices_globales,
                "posicion_mezclada": posiciones[pieza.id],
            }
            for pieza in piezas
        ],
    }
    (directorio_salida / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {"ancho": ancho, "alto": alto, "cantidad_piezas": len(piezas)}
