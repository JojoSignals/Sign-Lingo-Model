from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from src.inference.service import predict_letter, verify_letter

app = FastAPI(
    title="ASL Letter Recognition API",
    description="API para reconocer letras en lenguaje de señas (ASL)",
    version="1.0.0"
)

# Permitir CORS si vas a consumir desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # cámbialo por tu dominio si lo deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("api/v1/predict-letter")
async def predict(file: UploadFile = File(...)):
    """
    Predice la letra de la imagen cargada.
    """
    result = predict_letter(file.file)
    return result

@app.post("api/v1/verify-letter")
async def verify(
        file: UploadFile = File(...),
        expected_letter: str = Form(...)
):
    """
    Verifica si la imagen corresponde a la letra esperada.
    """
    result = verify_letter(file.file, expected_letter)
    return result
