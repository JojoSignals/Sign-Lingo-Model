# record_videos_by_word.py
import cv2
import os

def grabar_videos(palabra, num_videos=5, duracion_segundos=2, carpeta_base="data/raw/Videos"):
    carpeta_destino = os.path.join(carpeta_base, palabra)
    os.makedirs(carpeta_destino, exist_ok=True)

    # Contar cuántos videos ya hay para numerar los nuevos correctamente
    existentes = [f for f in os.listdir(carpeta_destino) if f.endswith('.mp4')]
    contador_inicio = len(existentes) + 1

    for i in range(num_videos):
        nombre_archivo = f"{palabra}_{str(contador_inicio + i).zfill(2)}.mp4"
        ruta_salida = os.path.join(carpeta_destino, nombre_archivo)

        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20
        total_frames = duracion_segundos * fps

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(ruta_salida, fourcc, fps, (width, height))

        print(f"🎬 Grabando {nombre_archivo}... (duración: {duracion_segundos} segundos)")

        frame_count = 0
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1

            # Mostrar la cámara en tiempo real con el conteo
            cv2.putText(frame, f"Grabando {palabra} ({i+1}/{num_videos})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Grabación', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"✅ Guardado en {ruta_salida}")

if __name__ == "__main__":
    palabra = input("🔤 Ingresa la palabra que estás grabando: ").strip().lower()
    num_videos = int(input("📦 ¿Cuántos videos deseas grabar? "))
    grabar_videos(palabra, num_videos=num_videos)
