import pickle, tensorflow as tf
model = tf.keras.models.load_model("models/sign_model.keras")
le = pickle.load(open("models/label_encoder.pkl", "rb"))

assert model.output_shape[-1] == len(le.classes_), (
    "Encoder y modelo no encajan: revisa los archivos")
print("Todo OK →", len(le.classes_), "clases")
