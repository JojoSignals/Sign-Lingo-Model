from keras.models import load_model

MODEL_PATH = "models/asl_letters_model.keras"
MODEL_FINETUNED = "models/asl_letters_finetuned.keras"
try:
    model = load_model(MODEL_FINETUNED)
    model.summary()
    print(f"✅ Modelo cargado exitosamente desde {MODEL_FINETUNED}")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")