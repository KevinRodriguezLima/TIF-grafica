from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.assembly.solver import solve_assembly
from src.compatibility.candidate_generator import generate_candidate_pairs
from src.compatibility.compatibility_scorer import filter_and_score_pairs
from src.compatibility.graph_builder import build_compatibility_graph
from src.content_classification.classifier import classify_reconstructed_image
from src.geometry.contour_extractor import extract_contours
from src.geometry.edge_extractor import detect_edges
from src.piece_classification.classifier import classify_pieces
from src.rendering.reconstructor import render_reconstruction
from src.segmentation.segmenter import build_pieces_manifest, segment_pieces


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pieces-dir", type=Path, default=Path("input/pieces"))
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--puzzle-id", default="puzzle_001")
    parser.add_argument("--alpha-threshold", type=int, default=10)
    parser.add_argument("--epsilon-factor", type=float, default=0.01)
    parser.add_argument("--min-edge-length", type=float, default=25.0)
    parser.add_argument("--strip-width", type=int, default=16)


def _add_late_stage_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ambiguous-threshold", type=float, default=0.50)
    parser.add_argument("--candidate-max-length-ratio", type=float, default=None)
    parser.add_argument("--filter-max-length-ratio", type=float, default=1.45)
    parser.add_argument("--max-opposite-angle-error", type=float, default=25.0)
    parser.add_argument("--min-compatibility-score", type=float, default=0.55)
    parser.add_argument("--top-k-per-edge", type=int, default=5)
    parser.add_argument("--piece-classifier-model", type=Path, default=None)
    parser.add_argument("--content-classifier-model", type=Path, default=None)
    parser.add_argument("--siamese-model", type=Path, default=None)
    parser.add_argument("--model-device", default="cpu")
    parser.add_argument("--grid-gap", type=int, default=24)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline de reconstruccion de imagenes fragmentadas")
    commands = parser.add_subparsers(dest="command", required=True)

    run_0_3 = commands.add_parser("run-stages-0-3", help="Ejecuta recepcion, segmentacion, contornos y lados")
    _add_common_arguments(run_0_3)

    run_0_5 = commands.add_parser("run-stages-0-5", help="Ejecuta desde recepcion hasta pares candidatos")
    _add_common_arguments(run_0_5)
    _add_late_stage_arguments(run_0_5)

    run_all = commands.add_parser("run-all", help="Ejecuta todas las etapas disponibles del pipeline")
    _add_common_arguments(run_all)
    _add_late_stage_arguments(run_all)
    return parser


def run_stages_0_3(args: argparse.Namespace) -> dict[str, Path]:
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
    contours = extract_contours(segmentation_path, contours_path, processed_dir / "contours")
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
        "results_dir": args.results_dir,
        "features_path": features_path,
    }


def run_stages_0_5(args: argparse.Namespace) -> dict[str, Path]:
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
        args.piece_classifier_model,
        args.model_device,
    )
    logging.info("Piezas clasificadas: %d", len(classification["pieces"]))

    logging.info("Etapa 5: generacion de pares candidatos")
    candidates = generate_candidate_pairs(
        features_path,
        classification_path,
        candidate_pairs_path,
        args.candidate_max_length_ratio,
    )
    logging.info("Bordes totales: %d", candidates["total_edges"])
    logging.info("Pares candidatos: %d", candidates["candidate_count"])
    paths.update({"classification_path": classification_path, "candidate_pairs_path": candidate_pairs_path})
    return paths


def run_all(args: argparse.Namespace) -> int:
    paths = run_stages_0_5(args)
    metadata_dir = paths["metadata_dir"]
    results_dir = paths["results_dir"]
    filtered_pairs_path = metadata_dir / "filtered_pairs.json"
    graph_path = metadata_dir / "compatibility_graph.json"
    assembly_path = results_dir / "assembly_solution.json"
    reconstructed_path = results_dir / "reconstructed.png"
    debug_path = results_dir / "reconstructed_debug.png"
    report_path = results_dir / "reconstruction_report.json"
    classification_result_path = results_dir / "classification_result.json"

    logging.info("Etapa 6 y 7: filtro geometrico y puntuacion visual/siamesa")
    filtered = filter_and_score_pairs(
        paths["candidate_pairs_path"],
        filtered_pairs_path,
        args.filter_max_length_ratio,
        args.max_opposite_angle_error,
        args.min_compatibility_score,
        args.siamese_model,
        args.model_device,
    )
    logging.info("Pares filtrados: %d", filtered["filtered_count"])

    logging.info("Etapa 8: grafo de compatibilidad")
    graph = build_compatibility_graph(filtered_pairs_path, paths["classification_path"], graph_path, args.top_k_per_edge)
    logging.info("Aristas del grafo: %d", len(graph["edges"]))

    logging.info("Etapa 9: ensamblaje")
    assembly = solve_assembly(graph_path, paths["features_path"], assembly_path, args.grid_gap)
    logging.info("Piezas colocadas: %d", assembly["pieces_used"])

    logging.info("Etapa 10: renderizado de reconstruccion")
    report = render_reconstruction(assembly_path, paths["features_path"], reconstructed_path, debug_path, report_path)
    logging.info("Imagen reconstruida: %s x %s", report["width"], report["height"])

    logging.info("Etapa 11: clasificacion del contenido")
    result = classify_reconstructed_image(
        reconstructed_path,
        classification_result_path,
        model_path=args.content_classifier_model,
        device=args.model_device,
    )
    logging.info("Prediccion aceptada: %s", result["accepted_prediction"])
    logging.info("Metadata generada en: %s", metadata_dir)
    logging.info("Resultados generados en: %s", results_dir)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    try:
        if args.command == "run-stages-0-3":
            paths = run_stages_0_3(args)
            logging.info("Metadata generada en: %s", paths["metadata_dir"])
            logging.info("Archivos procesados en: %s", paths["processed_dir"])
            return 0
        if args.command == "run-stages-0-5":
            paths = run_stages_0_5(args)
            logging.info("Metadata generada en: %s", paths["metadata_dir"])
            logging.info("Archivos procesados en: %s", paths["processed_dir"])
            return 0
        if args.command == "run-all":
            return run_all(args)
    except (FileNotFoundError, ValueError, OSError, ImportError) as error:
        logging.error("Error: %s", error)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
