import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.content_classification.dataset import CLASS_NAMES, inspect_dataset
from src.content_classification.model import IMAGE_SIZE, build_mobilenetv2, compile_model


def load_datasets(dataset_dir, batch_size, seed):
    inspect_dataset(dataset_dir)

    common = {
        "labels": "inferred",
        "label_mode": "int",
        "class_names": list(CLASS_NAMES),
        "image_size": IMAGE_SIZE,
        "batch_size": batch_size,
    }

    train_data = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "train",
        shuffle=True,
        seed=seed,
        **common,
    )
    validation_data = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "validation",
        shuffle=False,
        **common,
    )
    test_data = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "test",
        shuffle=False,
        **common,
    )

    train_data = train_data.prefetch(tf.data.AUTOTUNE)
    validation_data = validation_data.prefetch(tf.data.AUTOTUNE)
    test_data = test_data.prefetch(tf.data.AUTOTUNE)
    return train_data, validation_data, test_data


def train(
    dataset_dir,
    output_dir,
    epochs,
    batch_size,
    learning_rate,
    patience,
    seed,
    weights,
):
    if epochs < 1:
        raise ValueError("epochs debe ser mayor que cero")
    if batch_size < 1:
        raise ValueError("batch_size debe ser mayor que cero")
    if patience < 0:
        raise ValueError("patience no puede ser negativo")

    tf.keras.utils.set_random_seed(seed)
    train_data, validation_data, test_data = load_datasets(
        dataset_dir,
        batch_size,
        seed,
    )

    model, _ = build_mobilenetv2(weights=weights)
    compile_model(model, learning_rate)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=[early_stopping],
        shuffle=False,
    )
    test_metrics = model.evaluate(test_data, return_dict=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "mobilenetv2_classifier.keras"
    labels_path = output_dir / "labels.json"
    results_path = output_dir / "training_results.json"

    model.save(model_path)
    labels_path.write_text(
        json.dumps(list(CLASS_NAMES), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    results = {
        "model_path": str(model_path.as_posix()),
        "class_names": list(CLASS_NAMES),
        "epochs_completed": len(history.epoch),
        "test_metrics": {
            name: float(value)
            for name, value in test_metrics.items()
        },
        "history": {
            name: [float(value) for value in values]
            for name, values in history.history.items()
        },
    }
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return results


def build_parser():
    parser = argparse.ArgumentParser(
        description="Entrena MobileNetV2 para las cuatro clases del proyecto"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/content_classification"),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
    )
    return parser


def main():
    args = build_parser().parse_args()
    weights = None if args.weights == "none" else args.weights

    try:
        results = train(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            patience=args.patience,
            seed=args.seed,
            weights=weights,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
