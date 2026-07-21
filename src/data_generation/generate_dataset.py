import argparse
import logging
from pathlib import Path

from src.data_generation.image_downloader import descargar_imagenes_openverse
from src.data_generation.puzzle_generator import generar_rompecabezas


def _agregar_argumentos_generacion(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument(
        "--sides",
        type=int,
        choices=[0, 3, 4, 6],
        default=4,
        help="Lados por pieza (0 = Voronoi irregular)",
    )
    parser.add_argument("--seed", type=int, default=42)


def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generador de rompecabezas")
    comandos = parser.add_subparsers(dest="comando", required=True)

    generar = comandos.add_parser("generate", help="Divide y mezcla una imagen")
    generar.add_argument("--input", type=Path, required=True)
    _agregar_argumentos_generacion(generar)
    generar.add_argument("--output", type=Path, required=True)

    descargar = comandos.add_parser(
        "download-images", help="Busca y descarga imagenes con licencia abierta"
    )
    descargar.add_argument("--query", required=True, help="Texto de busqueda")
    descargar.add_argument("--limit", type=int, default=5)
    descargar.add_argument("--output", type=Path, default=Path("data/raw/downloaded"))

    lote = comandos.add_parser(
        "generate-batch", help="Genera rompecabezas para todas las imagenes de una carpeta"
    )
    lote.add_argument("--input-dir", type=Path, required=True)
    _agregar_argumentos_generacion(lote)
    lote.add_argument("--output", type=Path, required=True)

    web = comandos.add_parser(
        "generate-web", help="Descarga imagenes de internet y genera data en lote"
    )
    web.add_argument("--query", required=True, help="Texto de busqueda")
    web.add_argument("--limit", type=int, default=5)
    web.add_argument("--raw-output", type=Path, default=Path("data/raw/downloaded"))
    _agregar_argumentos_generacion(web)
    web.add_argument("--output", type=Path, required=True)
    return parser


def _generar_uno(argumentos: argparse.Namespace) -> dict[str, int]:
    return generar_rompecabezas(
        argumentos.input,
        argumentos.rows,
        argumentos.cols,
        argumentos.sides,
        argumentos.seed,
        argumentos.output,
    )


def _imagenes_en_carpeta(carpeta: Path) -> list[Path]:
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(ruta for ruta in carpeta.iterdir() if ruta.suffix.lower() in extensiones)


def _generar_desde_imagenes(
    imagenes: list[Path],
    carpeta_salida: Path,
    filas: int,
    columnas: int,
    lados: int,
    semilla: int,
) -> list[dict[str, object]]:
    resultados = []
    for indice, ruta_imagen in enumerate(imagenes, start=1):
        salida_item = carpeta_salida / f"{indice:03d}_{ruta_imagen.stem}"
        resultado = generar_rompecabezas(
            ruta_imagen,
            filas,
            columnas,
            lados,
            semilla + indice - 1,
            salida_item,
        )
        resultados.append({"imagen": str(ruta_imagen), "salida": str(salida_item), **resultado})
    return resultados


def _generar_lote(
    carpeta_entrada: Path,
    carpeta_salida: Path,
    filas: int,
    columnas: int,
    lados: int,
    semilla: int,
) -> list[dict[str, object]]:
    if not carpeta_entrada.is_dir():
        raise FileNotFoundError(f"No existe la carpeta: {carpeta_entrada}")

    imagenes = _imagenes_en_carpeta(carpeta_entrada)
    if not imagenes:
        raise ValueError(f"No se encontraron imagenes en: {carpeta_entrada}")
    return _generar_desde_imagenes(imagenes, carpeta_salida, filas, columnas, lados, semilla)


# Ejecuta el generador y mostramos un resumen.
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argumentos = crear_parser().parse_args()

    try:
        if argumentos.comando == "generate":
            resultado = _generar_uno(argumentos)
            logging.info("Rompecabezas generado correctamente.")
            logging.info("Imagen procesada: %d x %d", resultado["ancho"], resultado["alto"])
            logging.info("Cuadricula: %d x %d", argumentos.rows, argumentos.cols)
            logging.info("Lados por pieza: %d", argumentos.sides)
            logging.info("Piezas: %d", resultado["cantidad_piezas"])
            logging.info("Salida: %s", argumentos.output)
            return 0

        if argumentos.comando == "download-images":
            descargadas = descargar_imagenes_openverse(
                argumentos.query, argumentos.limit, argumentos.output
            )
            logging.info("Imagenes descargadas: %d", len(descargadas))
            logging.info("Salida: %s", argumentos.output)
            return 0 if descargadas else 1

        if argumentos.comando == "generate-batch":
            resultados = _generar_lote(
                argumentos.input_dir,
                argumentos.output,
                argumentos.rows,
                argumentos.cols,
                argumentos.sides,
                argumentos.seed,
            )
            logging.info("Datasets generados: %d", len(resultados))
            logging.info("Salida: %s", argumentos.output)
            return 0

        if argumentos.comando == "generate-web":
            descargadas = descargar_imagenes_openverse(
                argumentos.query, argumentos.limit, argumentos.raw_output
            )
            if not descargadas:
                raise ValueError("No se pudo descargar ninguna imagen valida")
            imagenes = [argumentos.raw_output / item["archivo"] for item in descargadas]
            resultados = _generar_desde_imagenes(
                imagenes,
                argumentos.output,
                argumentos.rows,
                argumentos.cols,
                argumentos.sides,
                argumentos.seed,
            )
            logging.info("Imagenes descargadas: %d", len(descargadas))
            logging.info("Datasets generados: %d", len(resultados))
            logging.info("Imagenes raw: %s", argumentos.raw_output)
            logging.info("Salida: %s", argumentos.output)
            return 0
    except (FileNotFoundError, ValueError, OSError) as error:
        logging.error("Error: %s", error)
        return 1

    logging.error("Comando no reconocido")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
