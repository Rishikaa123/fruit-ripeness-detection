from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np
from io import BytesIO
from PIL import Image

HERE = Path(__file__).resolve().parent
model = load_model(str(HERE / "FR4model.h5"))


app = FastAPI()

# Allow the browser page to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load model (robust path) ===
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "FR4model.h5"   # make sure the filename matches exactly
model = load_model(str(MODEL_PATH))

# Label map
fruits_class = {
    0: 'fresh_apples',
    1: 'fresh_banana',
    2: 'fresh_oranges',
    3: 'rotten_apples',
    4: 'rotten_banana',
    5: 'rotten_oranges'
}

def preprocess_image(pil_img: Image.Image):
    img = pil_img.resize((224, 224))   # adjust if your model needs a different size
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    x = preprocess_image(image)
    preds = model.predict(x)
    class_id = int(np.argmax(preds))
    confidence = float(preds[0][class_id])

    return {"class": fruits_class[class_id], "confidence": round(confidence, 4)}

