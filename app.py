

import pickle
import time

from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from motor import MotorRecuperacion


# --------------------------------
# CARGAR MODELO
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

RUTA_MODELO = BASE_DIR / "datos" / "modelo.pkl"

motor = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global motor

    with open(RUTA_MODELO, "rb") as archivo:

        datos = pickle.load(archivo)

    motor = MotorRecuperacion(datos)

    print("Motor de recuperación cargado.")
    print("Documentos:", motor.N)

    yield

    motor = None


app = FastAPI(
    title="NFCorpus Information Retrieval API",
    description=(
        "Servicio REST para recuperación de documentos "
        "científicos utilizando TF-IDF, BM25 e híbrido."
    ),
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------
# MODELO DE PETICION
# --------------------------------

class SearchRequest(BaseModel):

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="Consulta del usuario"
    )

    model: Literal[
        "tfidf",
        "bm25",
        "hibrido"
    ] = "bm25"

    top_k: int = Field(
        default=10,
        ge=1,
        le=100
    )

    alpha: float = Field(
        default=0.5,
        ge=0,
        le=1
    )


# --------------------------------
# ENDPOINT HEALTH
# --------------------------------

@app.get("/health")
def health():

    return {

        "status": "ok" if motor is not None else "loading",

        "service": "NFCorpus Information Retrieval API",

        "models": [
            "tfidf",
            "bm25",
            "hibrido"
        ],

        "documents": motor.N if motor else 0

    }


# --------------------------------
# ENDPOINT SEARCH
# --------------------------------

@app.post("/search")
def search(request: SearchRequest):

    if motor is None:

        raise HTTPException(
            status_code=503,
            detail="El motor todavía no está disponible."
        )

    inicio = time.perf_counter()

    try:

        resultados = motor.buscar(
            query=request.query,
            modelo=request.model,
            top_k=request.top_k,
            alpha=request.alpha
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Error durante la recuperación: {str(e)}"
        )

    tiempo = time.perf_counter() - inicio

    return {

        "query": request.query,

        "model": request.model,

        "top_k": request.top_k,

        "total_results": len(resultados),

        "execution_time_ms": round(tiempo * 1000, 3),

        "results": resultados

    }
