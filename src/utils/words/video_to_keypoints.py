import cv2
import os
import pickle
import time
import mediapipe as mp
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
def flatten_landmarks(lm, n_points):
    """Devuelve un vector fijo de n_points*2 valores."""
    if lm:
        return [coord for p in lm.landmark for coord in (p.x, p.y)]
    return [0.0] * (n_points * 2)

def extract_keypoints(results):
    pose = flatten_landmarks(results.pose_landmarks, 33)    # 66
    lh   = flatten_landmarks(results.left_hand_landmarks, 21)   # 42
    rh   = flatten_landmarks(results.right_hand_landmarks, 21)  # 42
    # si quieres cara:
    # face = flatten_landmarks(results.face_landmarks, 468)     # 936
    return pose + lh + rh       # total fijo: 150 features
# ─────────────────────────────────────────────────────────────

def process_video(video_path, output_pkl_path, max_frames=30):
    start = time.time()
    cap = cv2.VideoCapture(video_path)
    with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1
    ) as holistic:
        seq = []
        while cap.isOpened() and len(seq) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            seq.append(extract_keypoints(holistic.process(image)))
        cap.release()

        # padding si faltan frames o seq vacío
        if not seq:
            # si no se detectó nada, llena con ceros de largo fijo 150
            seq = [[0.0]*150 for _ in range(max_frames)]
        else:
            while len(seq) < max_frames:
                seq.append([0.0] * len(seq[0]))

    with open(output_pkl_path, "wb") as f:
        pickle.dump(seq, f)
    duration = time.time() - start
    print(f"✅ Guardado en {output_pkl_path} (procesado en {duration:.2f}s)")
    return duration

def process_batch(in_dir, out_dir, max_frames=30, start_label="lentes"):
    os.makedirs(out_dir, exist_ok=True)
    video_count = 0
    total_time = 0.0
    started = False

    for word in sorted(os.listdir(in_dir)):
        wd = os.path.join(in_dir, word)
        if not os.path.isdir(wd):
            continue

        # hasta encontrar la carpeta de inicio, saltar
        if not started:
            if word != start_label:
                print(f"⏭️  Saltando carpeta '{word}'")
                continue
            print(f"▶️  Comenzando desde carpeta '{word}'")
            started = True

        out_wd = os.path.join(out_dir, word)
        os.makedirs(out_wd, exist_ok=True)

        for file in tqdm(sorted(os.listdir(wd)), desc=f"Procesando '{word}'"):
            if not file.lower().endswith((".mp4", ".avi")):
                continue

            input_path  = os.path.join(wd, file)
            output_path = os.path.join(out_wd, file.rsplit(".",1)[0] + ".pkl")

            # si ya existe el .pkl, saltar
            if os.path.exists(output_path):
                continue

            duration = process_video(input_path, output_path, max_frames)
            video_count += 1
            total_time  += duration

    if video_count:
        avg_time = total_time / video_count
        print(f"\n🏁 Procesados {video_count} videos en {total_time:.2f}s (avg {avg_time:.2f}s/video)")
    else:
        print("⚠️  No se encontraron videos nuevos para procesar.")

if __name__ == "__main__":
    process_batch("data/raw/Videos", "data/raw/Keypoints", max_frames=30, start_label="lentes")
