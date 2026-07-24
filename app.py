import argparse
from pathlib import Path

from src.content_classification.classifier import classify_reconstructed_image
from src.main import run_all
from src.rendering.reconstruction_evaluator import evaluate_reconstruction


DEFAULT_MODEL = Path(
    "models/content_classification_coco_500/mobilenetv2_classifier.keras"
)


def solve(args):
    puzzle_id = args.puzzle_id or args.pieces.name
    pipeline_args = argparse.Namespace(
        pieces_dir=args.pieces,
        processed_dir=Path("processed") / puzzle_id,
        metadata_dir=Path("metadata") / puzzle_id,
        results_dir=Path("results") / puzzle_id,
        puzzle_id=puzzle_id,
        alpha_threshold=10,
        epsilon_factor=0.01,
        min_edge_length=25.0,
        strip_width=16,
        ambiguous_threshold=0.50,
        candidate_max_length_ratio=None,
        filter_max_length_ratio=1.45,
        max_opposite_angle_error=25.0,
        min_compatibility_score=0.55,
        top_k_per_edge=5,
        piece_classifier_model=None,
        content_classifier_model=args.model,
        siamese_model=None,
        model_device="cpu",
        grid_gap=24,
    )
    result = run_all(pipeline_args)
    if result == 0:
        print(f"Reconstrucción: results/{puzzle_id}/reconstructed.png")
        print(
            "Clasificación: "
            f"results/{puzzle_id}/classification_result.json"
        )
    return result


def classify(args):
    output = args.output or Path("results") / (
        f"{args.image.stem}_classification.json"
    )
    result = classify_reconstructed_image(
        args.image,
        output,
        model_path=args.model,
    )
    prediction = result["accepted_prediction"]
    confidence = result["predictions"][0]["confidence"] * 100
    print(f"Clase: {prediction}")
    print(f"Confianza: {confidence:.2f} %")
    print(f"Resultado: {output}")
    return 0


def evaluate(args):
    output = args.output or Path("results/reconstruction_evaluation.json")
    result = evaluate_reconstruction(
        args.reference,
        args.reconstruction,
        output,
    )
    print(f"Similitud: {result['similarity'] * 100:.2f} %")
    print(f"Resultado: {output}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    solve_parser = commands.add_parser("solve")
    solve_parser.add_argument(
        "pieces",
        type=Path,
        nargs="?",
        default=Path("input/pieces/imagen1"),
    )
    solve_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    solve_parser.add_argument("--puzzle-id", default=None)
    solve_parser.set_defaults(run=solve)

    classify_parser = commands.add_parser("classify")
    classify_parser.add_argument("image", type=Path)
    classify_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    classify_parser.add_argument("--output", type=Path, default=None)
    classify_parser.set_defaults(run=classify)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("reference", type=Path)
    evaluate_parser.add_argument("reconstruction", type=Path)
    evaluate_parser.add_argument("--output", type=Path, default=None)
    evaluate_parser.set_defaults(run=evaluate)
    return parser


def main():
    args = build_parser().parse_args()
    try:
        return args.run(args)
    except (FileNotFoundError, ValueError, OSError, ImportError) as error:
        print(f"Error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
