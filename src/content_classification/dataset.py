import argparse
import json
from pathlib import Path


CLASS_NAMES = ("animales", "frutas", "objetos", "personas")
SPLIT_NAMES = ("train", "validation", "test")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def _image_files(directory):
    return (
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"No existe el dataset: {dataset_dir}")

    split_counts = {}
    errors = []

    for split_name in SPLIT_NAMES:
        split_dir = dataset_dir / split_name
        if not split_dir.is_dir():
            errors.append(f"Falta la particion: {split_dir}")
            continue

        actual_classes = sorted(path.name for path in split_dir.iterdir() if path.is_dir())
        expected_classes = sorted(CLASS_NAMES)
        if actual_classes != expected_classes:
            missing = sorted(set(expected_classes) - set(actual_classes))
            unexpected = sorted(set(actual_classes) - set(expected_classes))
            if missing:
                errors.append(f"{split_name}: faltan clases: {', '.join(missing)}")
            if unexpected:
                errors.append(f"{split_name}: clases no esperadas: {', '.join(unexpected)}")

        class_counts = {}
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                continue
            count = sum(1 for _ in _image_files(class_dir))
            class_counts[class_name] = count
            if count == 0:
                errors.append(f"{split_name}/{class_name}: no contiene imagenes")
        split_counts[split_name] = class_counts

    if errors:
        raise ValueError("Dataset invalido:\n- " + "\n- ".join(errors))

    total_images = sum(
        count
        for class_counts in split_counts.values()
        for count in class_counts.values()
    )
    return {
        "dataset_dir": str(dataset_dir.as_posix()),
        "class_names": list(CLASS_NAMES),
        "splits": split_counts,
        "total_images": total_images,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Valida el dataset de clasificacion antes del entrenamiento"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Ruta opcional para guardar el resumen JSON",
    )
    return parser


def main():
    args = build_parser().parse_args()
    try:
        summary = inspect_dataset(args.dataset_dir)
    except (FileNotFoundError, ValueError) as error:
        print(error)
        return 1

    output = json.dumps(summary, indent=2, ensure_ascii=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
