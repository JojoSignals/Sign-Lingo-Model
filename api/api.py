from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.inference.service import (
    predict_letter, verify_letter,   # letras
    predict_word,  verify_word      # palabras
)

app = FastAPI(
    title="ASL & Word Recognition API",
    description="Reconoce letras (imagen) y palabras (vídeo keypoints).",
    version="1.1.0"
)

# ───────── CORS ─────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # ajusta a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════ ENDPOINTS LETRAS ═════════════════════════════════════

@app.post("/api/v1/predict-letter")
async def api_predict_letter(file: UploadFile = File(...)):
    try:
        res = predict_letter(file.file)
        return res
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception:
        raise HTTPException(500, detail="Error interno")

@app.post("/api/v1/verify-letter")
async def api_verify_letter(
        file: UploadFile = File(...),
        expected_letter: str = Form(...)
):
    try:
        res = verify_letter(file.file, expected_letter)
        return res
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception:
        raise HTTPException(500, detail="Error interno")

# ══════════ ENDPOINTS PALABRAS (VÍDEO) ═══════════════════════════

@app.post("/api/v1/predict-word")
async def api_predict_word(file: UploadFile = File(...)):
    """
    Sube un vídeo corto (.mp4) de ~2 s; devuelve la palabra detectada.
    """
    try:
        res = predict_word(file.file)
        return res
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception:
        raise HTTPException(500, detail="Error interno")

@app.post("/api/v1/verify-word")
async def api_verify_word(
        file: UploadFile = File(...),
        expected_word: str = Form(...)
):
    """
    Verifica si el vídeo corresponde a la palabra `expected_word`.
    """
    try:
        res = verify_word(file.file, expected_word)
        return res
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception:
        raise HTTPException(500, detail="Error interno")


from typing import Literal

@app.post("/api/v1/predict")
async def api_predict(
        file: UploadFile = File(...),
        mode: Literal["auto", "letter", "word"] = Form("auto")
):
    """
    Detecta automáticamente el tipo de archivo (.jpg/.png o .mp4) o permite forzar el modo.
    """
    try:
        filename = file.filename.lower()

        # ───── Forzar modo explícito ─────
        if mode == "letter":
            return predict_letter(file.file)
        elif mode == "word":
            return predict_word(file.file)

        # ───── Modo automático según extensión ─────
        if filename.endswith((".jpg", ".jpeg", ".png")):
            return predict_letter(file.file)
        elif filename.endswith(".mp4"):
            return predict_word(file.file)
        else:
            raise HTTPException(400, detail="Formato no soportado. Usa .jpg, .png o .mp4")
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception:
        raise HTTPException(500, detail="Error interno")


@app.post("/api/v1/verify")
async def api_verify(
        file: UploadFile = File(...),
        expected: str = Form(...),
        mode: Literal["auto", "letter", "word"] = Form("auto")
):
    """
    Verifica si el archivo representa correctamente la letra o palabra esperada.
    Usa 'mode' para forzar tipo: 'letter' o 'word'.
    """
    try:
        filename = file.filename.lower()

        # ───── Forzar modo explícito ─────
        if mode == "letter":
            return verify_letter(file.file, expected)
        elif mode == "word":
            return verify_word(file.file, expected)

        # ───── Modo automático según extensión ─────
        if filename.endswith((".jpg", ".jpeg", ".png")):
            return verify_letter(file.file, expected)
        elif filename.endswith(".mp4"):
            return verify_word(file.file, expected)
        else:
            raise HTTPException(400, detail="Formato no soportado. Usa .jpg, .png o .mp4")
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception:
        raise HTTPException(500, detail="Error interno")


# ══════════ Healthcheck ═════════════════════════════════════════

@app.get("/api/v1/health")
def health_check():
    return {"status": "ok"}
