import os
import random
import string
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

INPUT_DIR = "data/raw/Data"  # Ajusta si tu carpeta base tiene otro nombre
TARGET_COUNT = 508

def augment_image(img: Image.Image) -> Image.Image:
    """Aplica una transformación aleatoria simple a una imagen PIL."""
    transform = random.choice([
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),                   # Espejado
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.7, 1.3)),  # Brillo
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.8, 1.2)),    # Contraste
        lambda x: ImageOps.expand(x, border=random.randint(5, 15), fill='black'), # Padding
        lambda x: x.rotate(random.uniform(-10, 10))                    # Rotación
    ])
    return transform(img)

for letter in string.ascii_uppercase:
    folder = os.path.join(INPUT_DIR, letter)
    if not os.path.exists(folder):
        print(f"❌ Carpeta no encontrada: {folder}")
        continue

    images = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    current_count = len(images)

    print(f"📁 {letter}: {current_count} imágenes")

    if current_count >= TARGET_COUNT:
        print(f"✅ Ya hay suficientes imágenes.")
        continue

    needed = TARGET_COUNT - current_count
    print(f"➕ Generando {needed} nuevas imágenes para {letter}...")

    for i in range(needed):
        base_file = random.choice(images)
        base_path = os.path.join(folder, base_file)
        img = Image.open(base_path).convert("RGB")
        augmented = augment_image(img)
        new_filename = f"{letter}_aug_{i}.jpg"
        augmented.save(os.path.join(folder, new_filename))

print("🎉 Dataset nivelado a 508 imágenes por clase.")
