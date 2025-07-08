import cv2, numpy as np, pickle
from collections import deque, Counter
import mediapipe as mp
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
MAX_FRAMES    = 30
MODEL_PATH    = "models/words/sign_model_words.h5"
ENCODER_PATH  = "models/words/runs/2025-07-05_165349/label_encoder.pkl"

LM_MIN_VALID  = 50          # descarta frames con <50 landmarks “buenos”

# Votación por cola estable
HOLD_N        = 8           # 8 frames idénticos (~0.3 s)
CONF_MIN      = 0.60        # confianza media mínima

# Carga modelo y encoder
model = load_model(MODEL_PATH, compile=False)
label_encoder = pickle.load(open(ENCODER_PATH, "rb"))

# Mediapipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, model_complexity=1)

def flat(lm, n):
    return [c for p in lm.landmark for c in (p.x, p.y)] if lm else [0.0]*(n*2)

def extract(res):
    return flat(res.pose_landmarks,33)+flat(res.left_hand_landmarks,21)+flat(res.right_hand_landmarks,21)

# Buffers
seq_buffer   = deque(maxlen=MAX_FRAMES)
pred_indices = deque(maxlen=30)
conf_scores  = deque(maxlen=30)

# Webcam loop
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok: break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); img.flags.writeable=False
    res = holistic.process(img)
    vec = extract(res)

    # —— filtro de calidad ——
    if sum(v != 0.0 for v in vec) < LM_MIN_VALID:
        continue                    # descarta frame
    seq_buffer.append(vec)

    text = "…"
    if len(seq_buffer) == MAX_FRAMES:
        seq = np.expand_dims(np.array(seq_buffer,np.float32),0)
        preds = model.predict(seq, verbose=0)[0]
        idx   = int(np.argmax(preds))
        conf  = float(preds[idx])

        # —— cola estable ——
        pred_indices.append(idx)
        conf_scores.append(conf)
        
        tail_idx  = list(pred_indices)[-HOLD_N:]
        tail_conf = list(conf_scores)[-HOLD_N:]
        
        if len(tail_idx) == HOLD_N and len(set(tail_idx)) == 1:
            avg_conf = np.mean(tail_conf)
            if avg_conf >= CONF_MIN:
                label = label_encoder.inverse_transform([tail_idx[-1]])[0]
                text  = f"{label} ({avg_conf*100:.0f}%)"


    disp = cv2.flip(frame,1)
    cv2.putText(disp, text, (10,50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("Real-Time Sign Recognition (q para salir)", disp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(); cv2.destroyAllWindows(); holistic.close()
