from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

import tensorflow as tf
import numpy as np
import json

from PIL import Image

app = FastAPI(
    title="Waste Classification API",
    version="1.0"
)

# =========================
# LOAD MODEL
# =========================

loaded_model = tf.saved_model.load("saved_model")
infer = loaded_model.signatures["serving_default"]

# =========================
# LOAD CLASS NAMES
# =========================

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# =========================
# RECYCLE INFORMATION
# =========================

recycle_info = {
    "B3": "Limbah berbahaya",
    "kaca": "Bisa didaur ulang",
    "kardus": "Bisa didaur ulang",
    "kertas": "Bisa didaur ulang",
    "logam": "Bisa didaur ulang",
    "medis": "Limbah medis",
    "Plastik": "Bisa didaur ulang"
}


# =========================
# ROOT ENDPOINT
# =========================

@app.get("/")
def home():
    return {
        "message": "Waste Classification API Running"
    }


# =========================
# PREDICTION ENDPOINT
# =========================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # VALIDASI FILE
        if not file.content_type.startswith("image/"):
            return JSONResponse(
                content={
                    "error": "File harus berupa gambar"
                },
                status_code=400
            )

        # LOAD IMAGE
        image = Image.open(file.file).convert("RGB")

        image = image.resize((224, 224))

        # PREPROCESS IMAGE
        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = (
            tf.keras.applications.mobilenet_v2.preprocess_input(
                image_array
            )
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # CONVERT TO TENSOR
        tensor = tf.convert_to_tensor(
            image_array
        )

        # PREDICTION
        prediction = infer(tensor)

        prediction = next(
            iter(prediction.values())
        ).numpy()[0]

        predicted_index = int(
            np.argmax(prediction)
        )

        predicted_class = class_names[
            predicted_index
        ]

        confidence = float(
            np.max(prediction)
        )

        return JSONResponse(
            content={
                "prediction": predicted_class,
                "confidence": round(
                    confidence * 100,
                    2
                ),
                "info": recycle_info.get(
                    predicted_class,
                    "Informasi tidak tersedia"
                )
            }
        )

    except Exception as e:

        return JSONResponse(
            content={
                "error": str(e)
            },
            status_code=500
        )