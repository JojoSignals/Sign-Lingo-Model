import os
from keras.models import load_model
from keras.optimizers import Adam
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Configuración
DATA_DIR = "data/raw/fine_tune"
MODEL_ORIGINAL = "models/asl_letters_model.keras"
MODEL_FINETUNED = "models/asl_letters_finetuned.keras"
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 16
EPOCHS = 15

# Cargar el modelo original
model = load_model(MODEL_ORIGINAL)

# Congelar todas las capas excepto la última
for layer in model.layers[:-1]:
    layer.trainable = False

# Compilar con tasa de aprendizaje baja
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Generadores de datos con solo imágenes nuevas de C y W
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Entrenamiento
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Guardar modelo refinado
model.save(MODEL_FINETUNED)
print(f"✅ Modelo ajustado guardado en: {MODEL_FINETUNED}")
