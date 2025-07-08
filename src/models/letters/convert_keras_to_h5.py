from tensorflow.keras.models import load_model

input_path = "models/words/runs/2025-07-05_165349/best_model.keras"
output_path = "models/words/sign_model_words.h5"

model = load_model(input_path, compile=False)
model.save(output_path)
print(f"Modelo guardado en: {output_path}")
