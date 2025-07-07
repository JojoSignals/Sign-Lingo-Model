import os

DATA_DIR = "data/raw/Data"  # Cambia si tu carpeta raíz es otra

def contar_imagenes_por_clase(data_dir):
    conteo = {}
    for clase in sorted(os.listdir(data_dir)):
        clase_path = os.path.join(data_dir, clase)
        if os.path.isdir(clase_path):
            archivos = [
                f for f in os.listdir(clase_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            conteo[clase] = len(archivos)
    return conteo

conteo = contar_imagenes_por_clase(DATA_DIR)

for clase, cantidad in conteo.items():
    print(f"{clase}: {cantidad} imágenes")

