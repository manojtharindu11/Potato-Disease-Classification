import os
from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

app = FastAPI()

MODEL_PATH = (Path(__file__).resolve().parent / ".." / "models" / "1.keras").resolve()
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

_MODEL_LOCK = Lock()
_MODEL: tf.keras.Model | None = None
_MODEL_LOAD_ERROR: str | None = None


class DenseCompat(tf.keras.layers.Dense):
    def __init__(self, *args, quantization_config=None, **kwargs):
        super().__init__(*args, **kwargs)


def _load_model_compat(model_path: Path) -> tf.keras.Model:
    base_kwargs = {"compile": False}
    compat_kwargs = {
        **base_kwargs,
        "custom_objects": {
            "Dense": DenseCompat,
            "keras.layers.Dense": DenseCompat,
        },
    }

    last_exc: Exception | None = None

    # First try a normal load (fast path).
    try:
        return tf.keras.models.load_model(str(model_path), **base_kwargs)
    except Exception as exc:
        last_exc = exc

    # Retry with a Dense compat override. Some saved models include
    # `quantization_config` in Dense config which older runtimes reject.
    try:
        return tf.keras.models.load_model(str(model_path), **compat_kwargs)
    except Exception as exc:
        last_exc = exc

    # If this Keras version supports safe_mode, disable it and retry.
    try:
        return tf.keras.models.load_model(str(model_path), safe_mode=False, **compat_kwargs)
    except TypeError as exc:
        # safe_mode not supported in this environment
        last_exc = exc
    except Exception as exc:
        last_exc = exc

    assert last_exc is not None
    raise last_exc


def get_model() -> tf.keras.Model | None:
    global _MODEL, _MODEL_LOAD_ERROR
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            _MODEL = _load_model_compat(MODEL_PATH)
            _MODEL_LOAD_ERROR = None
        except Exception as exc:  # keep server alive even if model can't load
            _MODEL = None
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
        "model_path": str(MODEL_PATH),
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