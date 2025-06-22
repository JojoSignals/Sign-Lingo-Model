from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

@app.post("/api/v1/predict-letter")
async def predict(file: UploadFile = File(...)):
    """
    Predice la letra de la imagen cargada.
    """
    try:
        print("⚙️ Recibiendo archivo:", file.filename)
        result = predict_letter(file.file)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno en el servidor")

@app.post("/api/v1/verify-letter")
async def verify(
        file: UploadFile = File(...),
        expected_letter: str = Form(...)
):
    """
    Verifica si la imagen corresponde a la letra esperada.
    """
    try:
        result = verify_letter(file.file, expected_letter)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error interno en el servidor")

@app.get("/api/v1/health")
def health_check():
    """
    Endpoint de salud para verificar si la API está viva.
    """
    return {"status": "ok"}