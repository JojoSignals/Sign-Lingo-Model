import cv2, os, time

# ------------ CONFIGURACIÓN ------------
CARPETA_BASE = "data/raw/Videos_fine_tune4"
VISTAS       = ["front", "left", "right"]
FPS          = 20
DURACION_S   = 2
PREP_SEC     = 5         # segundos de preparación
# ---------------------------------------

def next_index(dir_path, palabra, vista):
    if not os.path.exists(dir_path):
        return 1
    nums = [
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(dir_path)
        if f.startswith(f"{palabra}_{vista}_") and f.endswith(".mp4")
    ]
    return max(nums, default=0) + 1

def countdown(msg, seconds=3):
    """Muestra un contador en pantalla antes de grabar."""
    for i in range(seconds, 0, -1):
        print(f"{msg} {i}…")
        time.sleep(1)

def grabar_clip(palabra, vista, idx):
    os.makedirs(f"{CARPETA_BASE}/{palabra}", exist_ok=True)
    nombre = f"{palabra}_{vista}_{idx:02d}.mp4"
    ruta   = f"{CARPETA_BASE}/{palabra}/{nombre}"

    cap = cv2.VideoCapture(0)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(ruta, fourcc, FPS, (width, height))
    total_frames = FPS * DURACION_S

    print(f"🎬 Grabando {nombre} ({DURACION_S}s) — pulsa q para cancelar")
    frames = 0
    while frames < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frames += 1
        cv2.putText(frame,
                    f"{palabra}  {vista}  {frames}/{total_frames}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Grabación", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Guardado en {ruta}\n")

def main():
    palabra = input("🔤 Palabra a grabar: ").strip().lower()
    n_clips = int(input("📦 ¿Clips por vista? (ej. 5): "))

    for vista in VISTAS:
        print(f"\n=== Vista: {vista.upper()} ===")
        countdown("⏳ Prepárate…", PREP_SEC)
        idx_start = next_index(f"{CARPETA_BASE}/{palabra}", palabra, vista)
        for i in range(n_clips):
            grabar_clip(palabra, vista, idx_start + i)

if __name__ == "__main__":
    main()
