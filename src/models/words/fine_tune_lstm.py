"""
Fine-tuning multivista — versión mejorada
----------------------------------------
• Descongela las últimas 6 capas del modelo base.
• LR más alto + callbacks menos agresivos.
• Val_split = 0.10, batch 32.
"""

import os, glob, pickle, numpy as np, tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path

# ---------- Ajustes ----------
PKL_DIR   = "data/raw/pkl2"
MODEL_IN  = "models/words/runs/2025-07-04-multiview-v2/sign_model.keras"
ENCODER_IN= "models/words/runs/2025-07-04-multiview-v2/label_encoder.pkl"
RUN_DIR   = "models/words/runs/2025-07-04-multiview-v3"

EPOCHS      = 100
BATCH_SIZE  = 32
LR          = 3e-4
VAL_SPLIT   = 0.10
# -----------------------------

def load_dataset(split=VAL_SPLIT):
    X, y = [], []
    classes = sorted(os.listdir(PKL_DIR))
    cls2idx = {c: i for i, c in enumerate(classes)}

    for cls in classes:
        for pkl in glob.glob(f"{PKL_DIR}/{cls}/*.pkl"):
            X.append(pickle.load(open(pkl, "rb")))
            y.append(cls2idx[cls])

    X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
    idx  = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    n_val = int(len(X) * split)
    return (X[n_val:], y[n_val:]), (X[:n_val], y[:n_val]), classes

def main():
    (train_X, train_y), (val_X, val_y), classes = load_dataset()
    print(f"🔹 Train: {train_X.shape[0]}   🔹 Val: {val_X.shape[0]}   🔹 Clases: {len(classes)}")

    model = tf.keras.models.load_model(MODEL_IN)

    # --- descongela últimas 6 capas ---
    for layer in model.layers[-6:]:
        layer.trainable = True

    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    print("🔧 Parámetros entrenables:", trainable_params)

    model.compile(
        optimizer=Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
    ]

    model.fit(
        train_X, train_y,
        validation_split=VAL_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # --- Guardado ---
    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
    model.save(f"{RUN_DIR}/sign_model.keras")

    le = pickle.load(open(ENCODER_IN, "rb"))
    assert list(le.classes_) == classes, "⚠️ Desajuste de clases/encoder."
    pickle.dump(le, open(f"{RUN_DIR}/label_encoder.pkl", "wb"))

    print(f"✅ Modelo y encoder guardados en {RUN_DIR}")

if __name__ == "__main__":
    main()
