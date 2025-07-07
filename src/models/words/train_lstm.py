# train_lstm.py

import os
import pickle
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback

DATA_DIR    = "data/raw/Keypoints"
MAX_FRAMES  = 30
RANDOM_SEED = 42

class TimingCallback(Callback):
    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        print("⏱️  Inicio de entrenamiento")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start
        print(
            f"Epoch {epoch+1:02d} — {epoch_time:.2f}s  "
            f"loss: {logs['loss']:.4f}  acc: {logs['accuracy']:.4f}  "
            f"val_loss: {logs['val_loss']:.4f}  val_acc: {logs['val_accuracy']:.4f}"
        )

    def on_train_end(self, logs=None):
        total_time = time.time() - self.train_start
        print(f"✅ Entrenamiento completo en {total_time:.2f}s")


def load_data(data_dir):
    X_raw, y_raw = [], []
    labels = sorted(os.listdir(data_dir))

    for label in labels:
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".pkl"):
                path = os.path.join(folder, file)
                seq = pickle.load(open(path, "rb"))
                if len(seq) == MAX_FRAMES:
                    X_raw.append(seq)
                    y_raw.append(label)

    if not X_raw:
        raise ValueError("No hay secuencias cargadas. Verifica tus .pkl")

    # Filtrar secuencias con feature length uniforme
    feat_len = len(X_raw[0][0])
    X_clean, y_clean = [], []
    for seq, lab in zip(X_raw, y_raw):
        if all(len(frame) == feat_len for frame in seq):
            X_clean.append(seq)
            y_clean.append(lab)
        else:
            print(f"❌ Descarta '{lab}' por feature_len distinto ({[len(f) for f in seq][:3]})")

    X = np.array(X_clean)
    y = np.array(y_clean)
    print(f"✅ Cargadas {len(X)} muestras — cada una con shape {X.shape[1:]}")
    return X, y


# ─── Carga y preparación de datos ─────────────────────────────
X, y = load_data(DATA_DIR)

le    = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_cat
)

# ─── Construcción del modelo ─────────────────────────────────
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(MAX_FRAMES, X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ─── Entrenamiento con métricas de tiempo ────────────────────
timing_cb = TimingCallback()
earlystop = EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=[timing_cb, earlystop]
)

# ─── Guardar modelo y encoder ────────────────────────────────
os.makedirs("models", exist_ok=True)
model.save("models/sign_model.keras")
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("🎉 Modelo guardado en 'models/sign_model.keras' y encoder en 'models/label_encoder.pkl'")
