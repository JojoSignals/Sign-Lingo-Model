# fine_tune_multivista_v3.py  (versión con refuerzo selectivo)
"""
Fine-tune multivista v3
───────────────────────
• Descongela las 6 últimas capas del modelo base.
• Sobre-muestrea y aplica rot+flip SOLO a clases débiles
  (calor, cornudo, telefono, bien, lo siento).
• Validación estratificada (≥1 muestra / clase).
• Carpeta de salida versionada timestamp-vN.
"""

import os, glob, pickle, random, datetime, shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ───── Ajustes ────────────────────────────────────────────────
PKL_DIR   = "data/raw/pkl3"
MODEL_IN  = "models/words/runs/2025-07-05_165349/best_model.keras"
ENCODER_IN= "models/words/runs/2025-07-05_165349/label_encoder.pkl"
RUN_ROOT  = "models/words/runs"

EPOCHS      = 60
BATCH_SIZE  = 32
LR          = 1e-5            # paso fino
VAL_FRAC    = 0.10
SEED        = 42
# ──────────────────────────────────────────────────────────────

# 1 ─── Carga dataset y split estratificado ───────────────────
def load_all_samples():
    X, y = [], []
    classes = sorted(os.listdir(PKL_DIR))
    cls2idx = {c: i for i, c in enumerate(classes)}
    for cls in classes:
        for p in glob.glob(f"{PKL_DIR}/{cls}/*.pkl"):
            X.append(pickle.load(open(p, "rb")))
            y.append(cls2idx[cls])
    return np.array(X, np.float32), np.array(y, np.int32), classes

def stratified_split(X, y, frac):
    sss = StratifiedShuffleSplit(1, test_size=frac, random_state=SEED)
    tr, va = next(sss.split(X, y))
    return (X[tr], y[tr]), (X[va], y[va])

# 2 ─── Aumento solo para clases débiles ───────────────────────
def build_datasets(train_X, train_y, val_X, val_y, classes):
    WEAK_NAMES = ["calor", "cornudo", "telefono", "bien", "lo siento"]
    WEAK_IDX   = tf.constant([classes.index(n) for n in WEAK_NAMES], tf.int32)

    # --- sobre-muestreo (duplica 2× cada muestra débil) ---
    mask = np.isin(train_y, WEAK_IDX.numpy())
    train_X = np.concatenate([train_X, train_X[mask], train_X[mask]])
    train_y = np.concatenate([train_y, train_y[mask], train_y[mask]])

    def aug_rotate_flip(seq):
        angle = tf.random.uniform([], -0.17, 0.17)
        c, s  = tf.cos(angle), tf.sin(angle)
        R     = tf.stack([[c, -s], [s, c]])
        pts   = tf.reshape(seq, (-1, 75, 2)) - 0.5
        pts   = tf.einsum('ij,bkj->bki', R, pts) + 0.5
        # flip x con 50 %
        pts   = tf.where(tf.random.uniform([]) > .5,
                         tf.concat([1 - pts[..., :1], pts[..., 1:]], -1),
                         pts)
        return tf.reshape(pts, tf.shape(seq))

    def map_fn(x, y):
        x = tf.cond(tf.reduce_any(tf.equal(y, WEAK_IDX)),
                    lambda: aug_rotate_flip(x),
                    lambda: x)
        return x, y

    train_ds = (tf.data.Dataset.from_tensor_slices((train_X, train_y))
                .shuffle(len(train_X), seed=SEED)
                .map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds   = (tf.data.Dataset.from_tensor_slices((val_X,  val_y))
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))
    return train_ds, val_ds

# 3 ─── Main ───────────────────────────────────────────────────
def main():
    tf.random.set_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    X, y, classes = load_all_samples()
    (trX, trY), (vaX, vaY) = stratified_split(X, y, VAL_FRAC)
    print(f"🔹 Train {len(trX)}  🔹 Val {len(vaX)}  🔹 Clases {len(classes)}")

    train_ds, val_ds = build_datasets(trX, trY, vaX, vaY, classes)

    model = tf.keras.models.load_model(MODEL_IN)
    for layer in model.layers[-6:]:
        layer.trainable = True

    model.compile(optimizer=Adam(LR),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    run_dir = Path(RUN_ROOT) / datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    i = 1
    while run_dir.exists():
        run_dir = run_dir.with_name(run_dir.name + f"-v{i}"); i += 1
    run_dir.mkdir(parents=True)
    shutil.copy2(ENCODER_IN, run_dir / "label_encoder.pkl")

    cbs = [ModelCheckpoint(str(run_dir/"best_model.keras"), save_best_only=True, verbose=1),
           ReduceLROnPlateau(factor=.5, patience=4, verbose=1),
           EarlyStopping(patience=8, restore_best_weights=True, verbose=1)]

    model.fit(train_ds, validation_data=val_ds,
              epochs=EPOCHS, callbacks=cbs, verbose=1)

    model.save(str(run_dir / "sign_model_final.keras"))
    print(f"✅ Artefactos guardados en {run_dir}")

if __name__ == "__main__":
    main()
