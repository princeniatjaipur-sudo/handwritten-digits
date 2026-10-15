"""Train and evaluate a simple MNIST multilayer perceptron."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# Training can run on machines without a graphical display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "digit_mlp.keras"
HISTORY_PATH = ARTIFACTS_DIR / "training_history.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
ACCURACY_PLOT_PATH = ARTIFACTS_DIR / "training_accuracy.png"
LOSS_PLOT_PATH = ARTIFACTS_DIR / "training_loss.png"
SAMPLES_PLOT_PATH = ARTIFACTS_DIR / "sample_predictions.png"


def build_model() -> tf.keras.Model:
    """Create the required feedforward network for 28-by-28 grayscale images."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28), name="image"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(128, activation="relu", name="hidden_1"),
            tf.keras.layers.Dense(64, activation="relu", name="hidden_2"),
            tf.keras.layers.Dense(10, activation="softmax", name="digit_probabilities"),
        ],
        name="mnist_mlp",
    )


def save_training_plots(history: tf.keras.callbacks.History) -> None:
    """Write separate accuracy and loss charts from the actual training run."""
    values = history.history
    epochs = range(1, len(values["accuracy"]) + 1)

    for metric, title, filename in (
        ("accuracy", "Model accuracy", ACCURACY_PLOT_PATH),
        ("loss", "Model loss", LOSS_PLOT_PATH),
    ):
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values[metric], marker="o", label=f"Training {metric}")
        validation_key = f"val_{metric}"
        if validation_key in values:
            plt.plot(epochs, values[validation_key], marker="o", label=f"Validation {metric}")
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()


def save_sample_predictions(
    model: tf.keras.Model, test_images: np.ndarray, test_labels: np.ndarray
) -> None:
    """Plot a small set of test images with their true and predicted digits."""
    images = test_images[:10]
    labels = test_labels[:10]
    probabilities = model.predict(images, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    figure, axes = plt.subplots(2, 5, figsize=(11, 5))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index], cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"Actual: {labels[index]} | Predicted: {predictions[index]}")
        axis.axis("off")
    figure.suptitle("Sample predictions from the MNIST test set")
    figure.tight_layout()
    figure.savefig(SAMPLES_PLOT_PATH, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs (default: 8).")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size (default: 128).")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must both be positive integers")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    # Keras downloads MNIST the first time and caches it for later runs.
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        train_images,
        train_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        shuffle=True,
        verbose=2,
    )
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
    model.save(MODEL_PATH)

    # Persist measured values so they can be inspected without inventing results.
    history_values = {key: [float(value) for value in series] for key, series in history.history.items()}
    HISTORY_PATH.write_text(json.dumps(history_values, indent=2), encoding="utf-8")
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "training_examples": int(len(train_images)),
        "validation_examples_per_epoch": int(len(train_images) * 0.1),
        "test_examples": int(len(test_images)),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_training_plots(history)
    save_sample_predictions(model, test_images, test_labels)

    print("\nMeasured test results from this run:")
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model and charts in: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
