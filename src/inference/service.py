import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageOps
import string
import cv2

# Ruta al modelo entrenado
MODEL_PATH = "models/asl_letters_finetuned_tf"

# Tamaño esperado por el modelo
IMAGE_SIZE = (300, 300)

# Lista de clases: A-Z
CLASSES = list(string.ascii_uppercase)

# Cargar modelo una sola vez
model = load_model(MODEL_PATH)

def preprocess_image(file) -> np.ndarray:
    print("🟢 Entrando a preprocess_image")
    try:
        file.seek(0)
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("No se pudo decodificar la imagen")

        # Redimensionar directamente a 300x300 sin recorte
        imgResize = cv2.resize(img, IMAGE_SIZE)
        imgNormalized = imgResize / 255.0
        img_array = np.expand_dims(imgNormalized, axis=0)
        return img_array
    except Exception as e:
        print("❌ Error en preprocess_image:", str(e))
        raise ValueError(f"Fallo en el preprocesamiento: {str(e)}")

def predict_letter(file) -> dict:
    print("🔵 Entrando a predict_letter")
    img_array = preprocess_image(file)

    preds = model.predict(img_array)[0]
    class_idx = np.argmax(preds)
    confidence = float(preds[class_idx])
    predicted_letter = CLASSES[class_idx]

    return {
        "predicted_letter": predicted_letter,
        "confidence": confidence
    }

def verify_letter(file, expected_letter: str) -> dict:
    result = predict_letter(file)
    is_correct = result["predicted_letter"].upper() == expected_letter.upper()

    return {
        "expected_letter": expected_letter.upper(),
        "predicted_letter": result["predicted_letter"],
        "confidence": result["confidence"],
        "is_correct": is_correct
    }
