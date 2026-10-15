# Handwritten Digit Recognition

A beginner-friendly handwritten digit recognition project using TensorFlow/Keras, the MNIST dataset, and a Streamlit web app. It trains a feedforward multilayer perceptron (MLP), evaluates it on MNIST's test split, saves real metrics and charts, and predicts digits in uploaded images.

## Project files

- `train.py` loads and normalizes MNIST, builds and trains the network, evaluates it, and saves all generated results under `artifacts/`.
- `app.py` presents the upload-and-predict Streamlit interface and converts uploaded images into MNIST-like inputs.
- `requirements.txt` lists the Python dependencies.
- `artifacts/` is created by training; it contains the saved model, measured metrics/history, and visualizations. It is intentionally not included until training has run.

## 1. Set up Python

Use Python 3.10–3.13. In the VS Code terminal, create and activate a virtual environment from this project folder (replace `3.11` with an installed supported version if needed):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use the terminal's Command Prompt profile, or run the environment's Python directly with `.venv\Scripts\python.exe`.

## 2. Train and evaluate

```powershell
python train.py
```

The first run downloads MNIST and may take several minutes. By default the network trains for 8 epochs with batches of 128. To change those settings, for example:

```powershell
python train.py --epochs 10 --batch-size 64
```

### What the training code does

1. Loads 60,000 MNIST training examples and 10,000 test examples.
2. Converts pixel intensities to floating-point values in the range 0–1.
3. Builds a Sequential network: Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax).
4. Trains with Adam and sparse categorical cross-entropy, reserving 10% of the training data for validation.
5. Evaluates the final model against the held-out test data and writes the actual measured test loss and accuracy.
6. Saves the model, training history, accuracy/loss charts, and examples of test predictions to `artifacts/`.

No accuracy or loss numbers are stated here in advance. See `artifacts/metrics.json` and the terminal output after running training for the results from your execution. Training results can vary with hardware, library versions, and training settings.

## 3. Start the web app

Train the model first, then launch Streamlit:

```powershell
streamlit run app.py
```

Open the local address Streamlit prints in the terminal. Upload a PNG, JPG, JPEG, BMP, or WebP image and select **Predict digit**. The app shows the original upload, the normalized 28×28 model input, the predicted digit, and the model's confidence estimate.

For best results, use a centered image containing one dark or light handwritten digit with little background clutter. The preprocessing converts it to grayscale, adjusts polarity to MNIST's white-on-black style, trims empty margins, resizes while preserving its aspect ratio, and places it on a 28×28 canvas. The UI reports invalid image files rather than crashing.

## Generated outputs

After a successful training run, `artifacts/` contains:

- `digit_mlp.keras` — trained model used by the web app.
- `metrics.json` — actual test accuracy/loss and run parameters.
- `training_history.json` — per-epoch training and validation measurements.
- `training_accuracy.png` and `training_loss.png` — learning curves.
- `sample_predictions.png` — actual/predicted labels for example test images.

Training is a separate step; launching the app before training will show a helpful missing-model message.
