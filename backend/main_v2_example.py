"""Example only: mount this router into your existing FastAPI app."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from .v2_api import router as v2_router
except ImportError:
    from v2_api import router as v2_router

app = FastAPI(title="Stock AI Determine v2")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(v2_router)


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "v2"}
