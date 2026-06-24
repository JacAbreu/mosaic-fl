"""
app.py — FastAPI application factory.

Registra os routers de domínio (prediction, patients, admin),
configura CORS, static files e o lifespan (startup/shutdown).
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import state
from .routers import admin, patients, prediction

logger = logging.getLogger(__name__)

_CORS_ORIGINS = os.getenv("FL_CORS_ORIGINS", "*").split(",")
_STATIC_DIR   = Path(__file__).parent / "static"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    await admin.startup_checks()
    engine = state._get_engine()
    if engine.checkpoint_path:
        logger.info("startup_checkpoint_loaded path=%s", engine.checkpoint_path)
    yield


app = FastAPI(
    title="MOSAIC-FL API",
    description=(
        "Módulo de Predição Federada para Possibilidades de Diagnóstico e Evoluções Clínicas. "
        "AI-based CDSS: o modelo estima probabilidades — a decisão clínica é sempre humana."
    ),
    version="0.1.0",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

app.include_router(prediction.router)
app.include_router(patients.router)
app.include_router(admin.router)


@app.get("/", include_in_schema=False)
async def dashboard():
    index = _STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "MOSAIC-FL API — painel web não encontrado em static/index.html"}
