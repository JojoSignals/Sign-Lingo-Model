# src/models/train.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.models.letters.baseline_model import create_asl_model

DATA_DIR = "data/raw/fine_tune"
MODEL_PATH = "models/letters/sign_model_v3.keras"
IMAGE_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 26

def main():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=False
    )

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

    model = create_asl_model((300, 300, 3), NUM_CLASSES)

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint(MODEL_PATH, save_best_only=True)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

    model.save(MODEL_PATH, save_format="tf")
    print(f"✅ Modelo guardado en: {MODEL_PATH}")
    
if __name__ == "__main__":
    main()
