import tensorflow as tf

from src.content_classification.dataset import CLASS_NAMES


IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (*IMAGE_SIZE, 3)


def build_mobilenetv2(
    weights="imagenet",
    dropout_rate=0.20,
):
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate debe estar entre 0.0 y 1.0")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="mobilenetv2_four_classes")
    return model, base_model


def compile_model(
    model,
    learning_rate=1e-3,
):
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser mayor que cero")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def model_metadata():
    return {
        "architecture": "MobileNetV2",
        "input_size": list(IMAGE_SIZE),
        "channels": 3,
        "class_names": list(CLASS_NAMES),
        "pretrained_weights": "imagenet",
    }
