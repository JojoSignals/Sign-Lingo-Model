"""
src/models/letters/fine_tune.py
Ajusta (fine-tuning) el modelo pre-entrenado de letras ASL
con nuevas imágenes —p. ej. más ejemplos de C y W—
sin olvidar las 24 letras restantes.
"""

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ─────────────────── Configuración ───────────────────────────
DATA_DIR         = "data/raw/data_train/val"           # 26 carpetas A-Z
MODEL_ORIGINAL   = "models/letters/sign_model_v3.keras"
MODEL_FINETUNED  = "models/letters/sign_model_v3_finetuned.keras"
IMAGE_SIZE       = (300, 300)
BATCH_SIZE       = 16
EPOCHS           = 30
LEARNING_RATE    = 1e-4

# ─────────────────── Cargar y preparar modelo ────────────────
model = load_model(MODEL_ORIGINAL)

# Congelar todas las capas excepto la última (26 salidas)
for layer in model.layers[:-1]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ─────────────────── Generadores de datos ────────────────────
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False        # aumenta variabilidad
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

# ─────────────────── Callbacks ───────────────────────────────
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_FINETUNED, save_best_only=True)
]

# ─────────────────── Entrenamiento ───────────────────────────
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ─────────────────── Guardar modelo ajustado ────────────────
model.save(MODEL_FINETUNED, save_format="tf")
print(f"✅ Modelo ajustado guardado en: {MODEL_FINETUNED}")
