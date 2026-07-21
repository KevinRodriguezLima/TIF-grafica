from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.compatibility.candidate_generator import generate_candidate_pairs
from src.geometry.contour_extractor import extract_contours
from src.geometry.edge_extractor import detect_edges
from src.piece_classification.classifier import classify_pieces
from src.segmentation.segmenter import build_pieces_manifest, segment_pieces


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pieces-dir", type=Path, default=Path("input/pieces"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--puzzle-id", default="puzzle_001")
    parser.add_argument("--alpha-threshold", type=int, default=10)
    parser.add_argument("--epsilon-factor", type=float, default=0.01)
    parser.add_argument("--min-edge-length", type=float, default=5.0)
    parser.add_argument("--strip-width", type=int, default=16)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline de reconstruccion de imagenes fragmentadas")
    commands = parser.add_subparsers(dest="command", required=True)

    run_0_3 = commands.add_parser("run-stages-0-3", help="Ejecuta recepcion, segmentacion, contornos y lados")
    _add_common_arguments(run_0_3)

    run_0_5 = commands.add_parser("run-stages-0-5", help="Ejecuta desde recepcion hasta pares candidatos")
    _add_common_arguments(run_0_5)
    run_0_5.add_argument("--ambiguous-threshold", type=float, default=0.50)
    run_0_5.add_argument(
        "--max-length-ratio",
        type=float,
        default=None,
        help="Filtro preliminar opcional para descartar pares con longitudes muy distintas",
    )
    return parser


def run_stages_0_3(args: argparse.Namespace) -> dict[str, object]:
    metadata_dir = args.metadata_dir
    processed_dir = args.processed_dir

    manifest_path = metadata_dir / "pieces_manifest.json"
    segmentation_path = metadata_dir / "segmentation.json"
    contours_path = metadata_dir / "contours.json"
    features_path = metadata_dir / "pieces_features.json"

    logging.info("Etapa 0: recepcion de piezas")
    manifest = build_pieces_manifest(args.pieces_dir, manifest_path, args.puzzle_id)
    logging.info("Piezas validas: %d", manifest["number_of_pieces"])

    logging.info("Etapa 1: segmentacion por canal alfa")
    segmentation = segment_pieces(
        manifest_path,
        processed_dir / "cropped",
        processed_dir / "masks",
        segmentation_path,
        args.alpha_threshold,
    )
    logging.info("Piezas segmentadas: %d", segmentation["processed_count"])

    logging.info("Etapa 2: extraccion de contornos")
    contours = extract_contours(
        segmentation_path,
        contours_path,
        processed_dir / "contours",
    )
    logging.info("Contornos extraidos: %d", len(contours["pieces"]))

    logging.info("Etapa 3: deteccion de lados")
    features = detect_edges(
        contours_path,
        features_path,
        processed_dir / "edge_strips",
        args.epsilon_factor,
        args.min_edge_length,
        args.strip_width,
    )
    logging.info("Piezas con lados detectados: %d", len(features["pieces"]))

    return {
        "metadata_dir": metadata_dir,
        "processed_dir": processed_dir,
        "features_path": features_path,
    }


def run_stages_0_5(args: argparse.Namespace) -> int:
    paths = run_stages_0_3(args)
    metadata_dir = paths["metadata_dir"]
    features_path = paths["features_path"]

    classification_path = metadata_dir / "piece_classification.json"
    candidate_pairs_path = metadata_dir / "candidate_pairs.json"

    logging.info("Etapa 4: clasificacion individual de piezas")
    classification = classify_pieces(
        features_path,
        classification_path,
        args.ambiguous_threshold,
    )
    logging.info("Piezas clasificadas: %d", len(classification["pieces"]))

    logging.info("Etapa 5: generacion de pares candidatos")
    candidates = generate_candidate_pairs(
        features_path,
        classification_path,
        candidate_pairs_path,
        args.max_length_ratio,
    )
    logging.info("Bordes totales: %d", candidates["total_edges"])
    logging.info("Pares candidatos: %d", candidates["candidate_count"])
    logging.info("Metadata generada en: %s", metadata_dir)
    logging.info("Archivos procesados en: %s", paths["processed_dir"])
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()

    try:
        if args.command == "run-stages-0-3":
            run_stages_0_3(args)
            logging.info("Metadata generada en: %s", args.metadata_dir)
            logging.info("Archivos procesados en: %s", args.processed_dir)
            return 0
        if args.command == "run-stages-0-5":
            return run_stages_0_5(args)
    except (FileNotFoundError, ValueError, OSError) as error:
        logging.error("Error: %s", error)
        return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
