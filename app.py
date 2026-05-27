from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import tensorflow as tf
import numpy as np
import json
import os

from io import BytesIO
from contextlib import asynccontextmanager
from typing import Any

from PIL import Image


# =========================
# PATH CONFIGURATION
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "saved_model")
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")


# Dibuat Any agar Pylance tidak menganggap infer selalu None
infer: Any = None
class_names: list[str] = []


# =========================
# RECYCLE INFORMATION
# =========================
recycle_info = {
    "B3": "Limbah B3. Perlu penanganan khusus karena berpotensi berbahaya bagi manusia dan lingkungan.",
    "Kaca": "Sampah kaca. Umumnya dapat didaur ulang jika dipisahkan dari sampah lain.",
    "Kardus": "Sampah kardus. Dapat didaur ulang jika dalam kondisi kering dan tidak terkontaminasi.",
    "Kertas": "Sampah kertas. Dapat didaur ulang jika tidak basah, berminyak, atau tercampur limbah lain.",
    "Logam": "Sampah logam. Umumnya dapat didaur ulang dan memiliki nilai guna kembali.",
    "Medis": "Limbah medis. Perlu penanganan khusus karena berisiko biologis atau infeksius.",
    "Plastik": "Sampah plastik. Sebagian jenis plastik dapat didaur ulang tergantung kode dan kondisinya.",
    "nonsampah": "Objek bukan sampah. Tidak perlu diklasifikasikan sebagai limbah.",
    "organik": "Sampah organik. Dapat diolah menjadi kompos atau dikelola melalui proses biodegradasi."
}


# =========================
# IMAGE PREPROCESSING
# =========================
def preprocess_mobilenet_v2(image_array: np.ndarray) -> np.ndarray:
    """
    Pengganti tf.keras.applications.mobilenet_v2.preprocess_input.

    MobileNetV2 memakai skala:
    pixel / 127.5 - 1

    Hasil akhir berada pada rentang -1 sampai 1.
    """
    return (image_array / 127.5) - 1.0


# =========================
# LOAD MODEL ON STARTUP
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global infer, class_names

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Folder model tidak ditemukan: {MODEL_PATH}")

    if not os.path.exists(CLASS_PATH):
        raise FileNotFoundError(f"File class_names.json tidak ditemukan: {CLASS_PATH}")

    loaded_model = tf.saved_model.load(MODEL_PATH)

    signatures = getattr(loaded_model, "signatures", None)

    if signatures is None:
        raise RuntimeError(
            "Model tidak memiliki signatures. Pastikan model diexport sebagai TensorFlow SavedModel."
        )

    if "serving_default" not in signatures:
        raise RuntimeError("Signature 'serving_default' tidak ditemukan pada model.")

    infer = signatures["serving_default"]

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        loaded_class_names = json.load(f)

    if not isinstance(loaded_class_names, list) or len(loaded_class_names) == 0:
        raise RuntimeError("Isi class_names.json harus berupa list dan tidak boleh kosong.")

    class_names = loaded_class_names

    if len(class_names) != 9:
        raise RuntimeError(
            f"Jumlah class_names harus 9 kelas, tetapi ditemukan {len(class_names)} kelas: {class_names}"
        )

    print("Model berhasil dimuat.")
    print(f"Class names: {class_names}")

    yield


# =========================
# FASTAPI APP
# =========================
app = FastAPI(
    title="Waste Classification API",
    version="1.0",
    lifespan=lifespan
)


# =========================
# CORS CONFIGURATION
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# ROUTES
# =========================
@app.get("/")
def home():
    return {
        "message": "Waste Classification API Running",
        "total_classes": len(class_names),
        "classes": class_names,
        "docs": "/docs",
        "predict_endpoint": "/predict",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": infer is not None,
        "total_classes": len(class_names),
        "classes": class_names
    }


def run_model_prediction(tensor: tf.Tensor) -> np.ndarray:
    """
    Menjalankan inference model.
    Dibuat terpisah agar lebih mudah dicek dan lebih aman untuk Pylance.
    """
    global infer

    if infer is None:
        raise RuntimeError("Model belum dimuat.")

    try:
        prediction_result = infer(tensor)
    except TypeError:
        input_signature = getattr(infer, "structured_input_signature", None)

        if input_signature is None:
            raise RuntimeError("Signature input model tidak ditemukan.")

        input_names = list(input_signature[1].keys())

        if not input_names:
            raise RuntimeError("Nama input model tidak ditemukan pada signature.")

        input_name = input_names[0]
        prediction_result = infer(**{input_name: tensor})

    prediction_tensor = list(prediction_result.values())[0]
    prediction = prediction_tensor.numpy()[0]

    return prediction


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        if infer is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Model belum siap. Coba beberapa saat lagi."}
            )

        content_type = file.content_type or ""

        if not content_type.startswith("image/"):
            return JSONResponse(
                status_code=400,
                content={"error": "File harus berupa gambar."}
            )

        image_bytes = await file.read()

        if not image_bytes:
            return JSONResponse(
                status_code=400,
                content={"error": "File gambar kosong."}
            )

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))

        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)

        image_array = preprocess_mobilenet_v2(image_array)

        tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)

        prediction = run_model_prediction(tensor)

        predicted_index = int(np.argmax(prediction))

        if predicted_index >= len(class_names):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Index prediksi melebihi jumlah class_names.",
                    "predicted_index": predicted_index,
                    "total_classes": len(class_names)
                }
            )

        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))

        return JSONResponse(
            content={
                "prediction": predicted_class,
                "confidence": round(confidence * 100, 2),
                "info": recycle_info.get(
                    predicted_class,
                    "Informasi tidak tersedia"
                )
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# =========================
# LOCAL RUN + RENDER COMPATIBLE
# =========================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )