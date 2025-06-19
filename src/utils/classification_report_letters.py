import os
import numpy as np
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuración
DATA_DIR = "data/raw/Data"
MODEL_PATH = "models/asl_letters_model.keras"
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 32

# Cargar modelo
model = load_model(MODEL_PATH)

# Generador de validación
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Predicciones
y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

# Reporte por clase
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)
