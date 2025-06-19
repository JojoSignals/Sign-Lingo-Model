import cv2
import os
import string
import mediapipe as mp
from datetime import datetime

BASE_DIR = "data/raw/custom_finetune"
os.makedirs(BASE_DIR, exist_ok=True)

for letter in string.ascii_uppercase:
    os.makedirs(os.path.join(BASE_DIR, letter), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
print("✋ Presiona una letra A-Z para capturar la mano recortada. ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Calcular bounding box
            h, w, _ = frame.shape
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
            y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)
            hand_crop = frame[y_min:y_max, x_min:x_max]

            # Mostrar solo la región de la mano
            cv2.imshow("Recorte de Mano", hand_crop)
    else:
        hand_crop = None
        cv2.imshow("Recorte de Mano", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    if 97 <= key <= 122 and hand_crop is not None:  # a-z
        letter = chr(key).upper()
        folder = os.path.join(BASE_DIR, letter)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"{letter}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, hand_crop)
        print(f"✅ Imagen recortada guardada en {folder}")

cap.release()
cv2.destroyAllWindows()
