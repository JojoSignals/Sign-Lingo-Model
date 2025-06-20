import os
import cv2
from collections import deque, Counter
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model

# Inicialización
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = load_model("models/asl_letters_model.keras")
imgSize = 300
offset = 26

# Etiquetas (orden alfabético)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
          'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Ventana temporal para estabilización
prediction_window = deque(maxlen=5)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        if imgCrop.size == 0:
            continue

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

        imgInput = cv2.resize(imgWhite, (300, 300)) / 255.0
        imgInput = imgInput.reshape(1, 300, 300, 3)
        prediction = model.predict(imgInput)
        index = np.argmax(prediction)
        confidence = round(100 * prediction[0][index], 2)

        prediction_window.append(index)
        most_common, count = Counter(prediction_window).most_common(1)[0]

        if count >= 3:
            label = labels[most_common]
            conf_label = round(100 * prediction[0][most_common], 2)
            label_text = f"{label} ({conf_label}%)" if conf_label >= 70 else f"{label} (?)"
        else:
            label_text = "Detectando..."

        cv2.rectangle(imgOutput, (x1, y1 - 70), (x1 + 400, y1 - 20), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label_text, (x1, y1 - 30),
                    cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (0, 255, 0), 4)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
