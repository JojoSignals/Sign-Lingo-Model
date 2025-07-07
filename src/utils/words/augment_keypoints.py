"""
Crea aumentos (rotación y flip) de los .pkl recién generados.
Guarda las copias junto al original:

    original.pkl
    original_rot.pkl
    original_flip.pkl
"""

import pickle, glob, numpy as np
from pathlib import Path
from tqdm import tqdm
import math, random

PKL_DIR   = "data/raw/Keypoints/pkl"   # <-- carpeta donde están TODOS los .pkl
MAX_FRAMES = 30                        # igual que tu modelo
N_FEAT     = 150                       # 75 puntos × 2 coords

# ----------------- helpers -----------------
def rotate_seq(seq, max_deg=15):
    """Rota levemente la secuencia en torno a (0.5,0.5)."""
    angle = math.radians(random.uniform(-max_deg, max_deg))
    c, s  = math.cos(angle), math.sin(angle)
    R     = np.array([[c, -s], [s, c]])

    out   = seq.copy()
    pts   = out.reshape(out.shape[0], -1, 2)
    pts  -= 0.5
    pts   = pts @ R.T
    pts  += 0.5
    return pts.reshape(out.shape).clip(0, 1)

def flip_seq(seq):
    out = seq.copy()
    out[:, 0::2] = 1 - out[:, 0::2]   # columnas x
    return out
# -------------------------------------------

def main():
    pkls = glob.glob(f"{PKL_DIR}/**/*.pkl", recursive=True)
    to_aug = []

    for p in pkls:
        stem = Path(p).stem
        rot_p  = Path(p).with_name(stem + "_rot.pkl")
        flip_p = Path(p).with_name(stem + "_flip.pkl")
        if not rot_p.exists() or not flip_p.exists():
            to_aug.append((p, rot_p, flip_p))

    if not to_aug:
        print("✅ Todos los .pkl ya tienen rotación y flip.")
        return

    for orig_p, rot_p, flip_p in tqdm(to_aug, desc="Augmentando keypoints"):
        seq = pickle.load(open(orig_p, "rb"))

        if not rot_p.exists():
            pickle.dump(rotate_seq(seq), open(rot_p, "wb"))
        if not flip_p.exists():
            pickle.dump(flip_seq(seq), open(flip_p, "wb"))

    print(f"✅ Generados aumentos para {len(to_aug)} secuencias")

if __name__ == "__main__":
    main()
