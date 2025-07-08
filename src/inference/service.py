import numpy as np, string, cv2, pickle, tempfile, os
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import mediapipe as mp

# ──────────────────────────────────────────────────────────────
#  BLOCK A –  LETRAS  (imagen .jpg)                            │
# ──────────────────────────────────────────────────────────────
# Ruta al modelo entrenado
MODEL_PATH = "models/letters/sign_model_letter.h5"

# Tamaño esperado por el modelo
IMAGE_SIZE = (300, 300)

# Lista de clases: A-Z
CLASSES = list(string.ascii_uppercase)

# Cargar modelo y detector una sola vez (evita reinicios por exceso de memoria)
model = load_model(MODEL_PATH)
detector = HandDetector(maxHands=1)
def preprocess_image(file) -> np.ndarray:
    """Detecta la mano en la imagen, la recorta y la ajusta al formato del modelo."""
    print("🟢 Entrando a preprocess_image")
    file.seek(0)
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    imgSize = 300
    offset = 26
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    hands, img = detector.findHands(img)
    if not hands:
        raise ValueError("No se detectó ninguna mano en la imagen")

    hand = hands[0]
    x, y, w, h = hand["bbox"]

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
        imgWhite[:, wGap : wGap + wCal] = imgResize
    else:
        k = imgSize / (x2 - x1)
        hCal = int(k * (y2 - y1))
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = int((imgSize - hCal) / 2)
        imgWhite[hGap : hGap + hCal, :] = imgResize

    imgWhite = imgWhite / 255.0
    return np.expand_dims(imgWhite, axis=0)


def predict_letter(file) -> dict:
    """Predice la letra más probable de una imagen .jpg (solo imagen original)."""
    print("🔵 Entrando a predict_letter")
    img_array = preprocess_image(file)

    preds = model.predict(img_array, verbose=0)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    predicted_letter = CLASSES[class_idx]

    return {
        "predicted_letter": predicted_letter,
        "confidence": confidence,
    }


def verify_letter(file, expected_letter: str) -> dict:
    """Verifica si la imagen corresponde a la letra esperada."""
    result = predict_letter(file)
    is_correct = result["predicted_letter"].upper() == expected_letter.upper()

    return {
        "expected_letter": expected_letter.upper(),
        **result,
        "is_correct": is_correct,
    }

# ──────────────────────────────────────────────────────────────
#  BLOCK B –  PALABRAS  (vídeo .mp4 de ~30 frames)             │
# ──────────────────────────────────────────────────────────────
WORD_MODEL_PATH   = "models/words/sign_model_words.h5"
ENCODER_PATH      = "models/words/label_encoder_words.pkl"
MAX_FRAMES, N_FEAT = 30, 150

word_model   = load_model(WORD_MODEL_PATH, compile=False)
label_encoder= pickle.load(open(ENCODER_PATH,"rb"))
mp_holistic  = mp.solutions.holistic
holistic     = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

def _flatten(lm,n): return [c for p in lm.landmark for c in (p.x,p.y)] if lm else [0.0]*(n*2)
def _video_to_seq(path:str)->np.ndarray:
    cap, seq = cv2.VideoCapture(path), []
    while len(seq)<MAX_FRAMES:
        ok, fr = cap.read()
        if not ok: break
        res = holistic.process(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
        seq.append(_flatten(res.pose_landmarks,33)+
                   _flatten(res.left_hand_landmarks,21)+
                   _flatten(res.right_hand_landmarks,21))
    cap.release()
    if len(seq)<MAX_FRAMES: seq+=[[0.0]*N_FEAT]*(MAX_FRAMES-len(seq))
    return np.array(seq[:MAX_FRAMES],np.float32)

def _bytes_to_temp(file)->str:
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
    file.seek(0); tmp.write(file.read()); tmp.close(); return tmp.name

def preprocess_video(file):
    path=_bytes_to_temp(file); seq=_video_to_seq(path); os.remove(path)
    return np.expand_dims(seq,0)

def predict_word(file):
    seq = preprocess_video(file)
    preds=word_model.predict(seq,verbose=0)[0]
    idx = int(np.argmax(preds))
    return {"predicted_word": label_encoder.inverse_transform([idx])[0],
            "confidence": float(preds[idx])}

def verify_word(file, expected_word:str):
    res = predict_word(file)
    return {"expected_word": expected_word,
            **res,
            "is_correct": res["predicted_word"].lower()==expected_word.lower()}