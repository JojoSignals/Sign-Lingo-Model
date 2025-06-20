import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image, ImageOps
import string
import cv2
from cvzone.HandTrackingModule import HandDetector


# Ruta al modelo entrenado
MODEL_PATH = "models/asl_letters_finetuned.keras"

# Tamaño esperado por el modelo
IMAGE_SIZE = (300, 300)

# Lista de clases: A-Z
CLASSES = list(string.ascii_uppercase)

# Cargar modelo una sola vez (para uso en API)
model = load_model(MODEL_PATH)

def preprocess_image(file) -> np.ndarray:
    """Detecta la mano en la imagen, la recorta y la ajusta al formato del modelo."""
    file.seek(0)
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    detector = HandDetector(maxHands=1)
    imgSize = 300
    offset = 26
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgH, imgW, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(imgW, x + w + offset)
        y2 = min(imgH, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size == 0:
            raise ValueError("Recorte de mano fallido")

        aspectRatio = (y2 - y1) / (x2 - x1)

        if aspectRatio > 1:
            k = imgSize / (y2 - y1)
            wCal = int(k * (x2 - x1))
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = int((imgSize - wCal) / 2)
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / (x2 - x1)
            hCal = int(k * (y2 - y1))
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = int((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        imgWhite = imgWhite / 255.0
        img_array = np.expand_dims(imgWhite, axis=0)
        return img_array
    else:
        raise ValueError("No se detectó ninguna mano en la imagen")


def predict_letter(file) -> dict:
    """Predice la letra más probable de una imagen .jpg"""
    def predict_letter(file) -> dict:
        """Predice la letra más probable de una imagen .jpg, evaluando también su versión espejada."""
    img_array = preprocess_image(file)

    # Generar imagen espejada horizontalmente
    mirrored_array = np.flip(img_array, axis=2)  # flip axis 2 = ancho

    # Predecir ambas versiones
    preds_original = model.predict(img_array)[0]
    preds_mirrored = model.predict(mirrored_array)[0]

    # Elegir la de mayor confianza global
    if np.max(preds_original) >= np.max(preds_mirrored):
        class_idx = np.argmax(preds_original)
        confidence = float(preds_original[class_idx])
    else:
        class_idx = np.argmax(preds_mirrored)
        confidence = float(preds_mirrored[class_idx])

    predicted_letter = CLASSES[class_idx]

    return {
        "predicted_letter": predicted_letter,
        "confidence": confidence
    }


def verify_letter(file, expected_letter: str) -> dict:
    """Verifica si la imagen corresponde a la letra esperada."""
    result = predict_letter(file)
    is_correct = result["predicted_letter"].upper() == expected_letter.upper()

    return {
        "expected_letter": expected_letter.upper(),
        "predicted_letter": result["predicted_letter"],
        "confidence": result["confidence"],
        "is_correct": is_correct
    }
