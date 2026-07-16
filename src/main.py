import argparse
import logging
from pathlib import Path

from src.puzzle_generator import generar_rompecabezas

def crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generador de rompecabezas")
    comandos = parser.add_subparsers(dest="comando", required=True)
    generar = comandos.add_parser("generate", help="Divide y mezcla una imagen")
    generar.add_argument("--input", type=Path, required=True)
    generar.add_argument("--rows", type=int, required=True)
    generar.add_argument("--cols", type=int, required=True)
    generar.add_argument("--seed", type=int, default=42)
    generar.add_argument("--output", type=Path, required=True)
    return parser

# Ejecuta el generador y mostramos un resumen.
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    argumentos = crear_parser().parse_args()

    try:
        resultado = generar_rompecabezas(
            argumentos.input,
            argumentos.rows,
            argumentos.cols,
            argumentos.seed,
            argumentos.output,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        logging.error("Error: %s", error)
        return 1

    logging.info("Rompecabezas generado correctamente.")
    logging.info("Imagen procesada: %d x %d", resultado["ancho"], resultado["alto"])
    logging.info("Cuadrícula: %d x %d", argumentos.rows, argumentos.cols)
    logging.info("Piezas: %d", resultado["cantidad_piezas"])
    logging.info("Salida: %s", argumentos.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
