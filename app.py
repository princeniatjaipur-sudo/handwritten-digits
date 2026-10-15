"""Streamlit interface for predicting a digit in an uploaded image."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "digit_mlp.keras"


@st.cache_resource
def load_model(model_path: str) -> tf.keras.Model:
    """Load the trained network once and reuse it across Streamlit reruns."""
    return tf.keras.models.load_model(model_path)


def prepare_image(image: Image.Image) -> tuple[Image.Image, np.ndarray]:
    """Convert an upload to a centered, MNIST-like 28x28 white-on-black image."""
    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale, dtype=np.uint8)

    # MNIST uses dark backgrounds. If most border pixels are light, flip polarity.
    border = np.concatenate((pixels[0, :], pixels[-1, :], pixels[:, 0], pixels[:, -1]))
    if float(np.median(border)) > 127:
        grayscale = ImageOps.invert(grayscale)

    # Remove empty border, preserve proportions, then center the digit on a 28x28 canvas.
    foreground = grayscale.point(lambda value: 255 if value > 30 else 0)
    bounding_box = foreground.getbbox()
    canvas = Image.new("L", (28, 28), color=0)
    if bounding_box:
        # Crop the original grayscale pixels to retain anti-aliased stroke edges.
        digit = grayscale.crop(bounding_box)
        digit.thumbnail((20, 20), Image.Resampling.LANCZOS)
        x_offset = (28 - digit.width) // 2
        y_offset = (28 - digit.height) // 2
        canvas.paste(digit, (x_offset, y_offset))

    model_input = np.asarray(canvas, dtype=np.float32) / 255.0
    model_input = model_input.reshape(1, 28, 28)
    return canvas, model_input


st.set_page_config(page_title="Handwritten Digit Recognizer", page_icon="✍️", layout="centered")
st.title("Handwritten Digit Recognizer")
st.write("Upload a clear image of one handwritten digit. The model predicts a digit from 0 to 9.")
st.caption("This demo uses an MLP trained on MNIST; unusual writing styles may produce lower confidence.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "bmp", "webp"])

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file)
        uploaded_image.load()
        st.image(uploaded_image, caption="Uploaded image", width=240)

        if st.button("Predict digit", type="primary"):
            if not MODEL_PATH.exists():
                st.error("The trained model is not available yet. Run `python train.py` first, then restart or refresh this app.")
            else:
                model = load_model(str(MODEL_PATH))
                processed_image, model_input = prepare_image(uploaded_image)
                probabilities = model.predict(model_input, verbose=0)[0]
                prediction = int(np.argmax(probabilities))
                confidence = float(probabilities[prediction])

                left, right = st.columns(2)
                with left:
                    st.image(processed_image, caption="28×28 model input", width=180)
                with right:
                    st.metric("Predicted digit", prediction)
                    st.metric("Confidence", f"{confidence:.1%}")
                st.caption("Confidence is the model's probability estimate, not a guarantee of correctness.")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        st.error(f"Could not read this image. Please upload a valid PNG, JPG, BMP, or WebP file. Details: {error}")
