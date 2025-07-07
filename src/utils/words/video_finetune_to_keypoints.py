"""
Extrae keypoints (pose + manos) de los vídeos NUEVOS que aún no tienen .pkl
y los guarda en data/raw/Keypoints/pkl/<palabra>/<nombre>.pkl

Salida de cada .pkl → numpy array   shape = (MAX_FRAMES, 150)
150 = 75 puntos (33 pose + 21 LH + 21 RH) × 2 coords (x,y)

NUEVO:
• normaliza cada secuencia: centra en hombro izquierdo y escala por
  distancia entre hombros ⇒ robusto a distancia-zoom.
"""

import cv2, mediapipe as mp, pickle, glob, os, numpy as np
from pathlib import Path
from tqdm import tqdm   # pip install tqdm

VIDEO_DIR = "data/raw/Videos_fine_tune4"
PKL_DIR   = "data/raw/pkl4"
MAX_FRAMES = 30         # igual que tu modelo
N_FEAT     = 150        # 75 × 2

# ---------- MediaPipe setup ----------
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
)
# -------------------------------------

def flatten_landmarks(lm, n_pts):
    if lm:
        return [c for p in lm.landmark for c in (p.x, p.y)]
    return [0.0] * (n_pts * 2)

# ───────── NORMALIZACIÓN NUEVA ─────────────────
def normalize(seq):
    """
    • Centra en hombro izquierdo (landmark 11).
    • Escala dividiendo por la distancia entre hombros (11 y 12).
    seq: ndarray (frames, 150)
    """
    pts = seq.reshape(seq.shape[0], -1, 2)       # (T, 75, 2)
    center = pts[:, 11:12, :]                    # hombro L
    shoulder_r = pts[:, 12:13, :]
    scale = np.linalg.norm(center - shoulder_r, axis=-1, keepdims=True) + 1e-6
    pts = (pts - center) / scale
    return pts.reshape(seq.shape)
# ───────────────────────────────────────────────

def video_to_seq(path):
    cap = cv2.VideoCapture(str(path))
    seq = []

    while len(seq) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        res = holistic.process(img_rgb)

        vec = (
                flatten_landmarks(res.pose_landmarks, 33)
                + flatten_landmarks(res.left_hand_landmarks, 21)
                + flatten_landmarks(res.right_hand_landmarks, 21)
        )
        seq.append(vec)

    cap.release()

    # padding / truncado
    if len(seq) < MAX_FRAMES:
        pad = [[0.0] * N_FEAT] * (MAX_FRAMES - len(seq))
        seq.extend(pad)
    else:
        seq = seq[:MAX_FRAMES]

    seq = np.array(seq, dtype=np.float32)
    seq = normalize(seq)          # <<--- normalización aquí
    return seq

def main():
    videos = glob.glob(f"{VIDEO_DIR}/**/*.mp4", recursive=True)
    new_videos = []

    for v in videos:
        cls = Path(v).parent.name
        name = Path(v).stem + ".pkl"
        out_p = Path(PKL_DIR) / cls / name
        if not out_p.exists():
            new_videos.append((v, out_p))

    if not new_videos:
        print("✅ No hay vídeos nuevos por procesar.")
        return

    for v, out_p in tqdm(new_videos, desc="Extrayendo keypoints"):
        out_p.parent.mkdir(parents=True, exist_ok=True)
        seq = video_to_seq(v)
        pickle.dump(seq, open(out_p, "wb"))

    print(f"✅ Convertidos {len(new_videos)} vídeos → .pkl")

if __name__ == "__main__":
    main()
