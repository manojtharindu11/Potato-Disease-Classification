import os
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
import keras

app = FastAPI()

MODEL_CANDIDATE_PATHS = [
    (Path(__file__).resolve().parent / ".." / "models" / "1.keras").resolve(),
    (Path(__file__).resolve().parent / ".." / "models" / "2.keras").resolve(),
]
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

_MODEL_LOCK = Lock()
_MODEL: tf.keras.Model | None = None
_MODEL_LOAD_ERROR: str | None = None
_MODEL_PATH_USED: str | None = None


class DenseCompat(tf.keras.layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


def _load_model_compat(model_path: Path) -> tf.keras.Model:
    base_kwargs = {"compile": False}

    # Keras deserialization sometimes ignores custom_objects for built-ins.
    # As a stronger workaround, temporarily patch Dense to a compat class that
    # accepts `quantization_config`.
    orig_keras_dense = keras.layers.Dense
    orig_tf_dense = tf.keras.layers.Dense
    try:
        keras.layers.Dense = DenseCompat  # type: ignore[assignment]
        tf.keras.layers.Dense = DenseCompat  # type: ignore[assignment]

        try:
            return tf.keras.models.load_model(str(model_path), **base_kwargs)
        except TypeError:
            # Some environments support safe_mode; try disabling if available.
            return tf.keras.models.load_model(str(model_path), safe_mode=False, **base_kwargs)
    finally:
        keras.layers.Dense = orig_keras_dense  # type: ignore[assignment]
        tf.keras.layers.Dense = orig_tf_dense  # type: ignore[assignment]


def _load_any_model() -> tuple[tf.keras.Model, str]:
    last_exc: Exception | None = None
    for candidate in MODEL_CANDIDATE_PATHS:
        if not candidate.exists():
            continue
        try:
            model = _load_model_compat(candidate)
            return model, str(candidate)
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc is not None:
        raise last_exc

    raise FileNotFoundError(
        "No model file found. Tried: " + ", ".join(str(p) for p in MODEL_CANDIDATE_PATHS)
    )


def get_model() -> tf.keras.Model | None:
    global _MODEL, _MODEL_LOAD_ERROR, _MODEL_PATH_USED
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            _MODEL, _MODEL_PATH_USED = _load_any_model()
            _MODEL_LOAD_ERROR = None
        except Exception as exc:  # keep server alive even if model can't load
            _MODEL = None
            _MODEL_PATH_USED = None
            _MODEL_LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        return _MODEL




@app.get("/ping")
async def ping():
    return "Hello I am alive.."


@app.get("/health")
async def health():
    model = get_model()
    return {
        "ok": True,
        "model_loaded": model is not None,
        "model_path": _MODEL_PATH_USED,
        "model_candidates": [str(p) for p in MODEL_CANDIDATE_PATHS],
        "model_error": _MODEL_LOAD_ERROR,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail={"error": "Model not loaded", "details": _MODEL_LOAD_ERROR})

    image = read_file_as_image(await file.read(), model)
    return image

def read_file_as_image(data, model: tf.keras.Model):
    image =  np.array(Image.open(BytesIO(data)))
    img_batch = np.expand_dims(image, 0)
    
    prediction = model.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)