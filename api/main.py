import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os
from pathlib import Path

app = FastAPI()

MODEL_PATH = (Path(__file__).resolve().parent / ".." / "models" / "1.keras").resolve()
MODEL = tf.keras.models.load_model(str(MODEL_PATH))
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello I am alive.."

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
   image = read_file_as_image(await file.read())
   return image

def read_file_as_image(data) -> np.ndarray:
    image =  np.array(Image.open(BytesIO(data)))
    img_batch = np.expand_dims(image, 0)
    
    prediction = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }
 
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)