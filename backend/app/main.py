"""
XI-Lab FastAPI application.

Endpoints (stubs — implemented as the ML pipeline is built):
  POST /recommend   → best XI + tie set + batting order + bowling plan + swap matrix
  POST /simulate    → win probability + score distributions for two XIs
  POST /compare     → paired ΔWP with confidence interval
  GET  /player/{id} → phase ratings + matchups + role + SHAP summary
  GET  /live        → prediction log with results
  GET  /results     → evaluation artifacts
  GET  /health      → liveness check
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import compare, live, player, recommend, results, simulate

app = FastAPI(
    title="XI-Lab",
    description="Context-aware cricket team selection engine",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, prefix="/recommend", tags=["selection"])
app.include_router(simulate.router, prefix="/simulate", tags=["simulation"])
app.include_router(compare.router, prefix="/compare", tags=["simulation"])
app.include_router(player.router, prefix="/player", tags=["players"])
app.include_router(live.router, prefix="/live", tags=["live"])
app.include_router(results.router, prefix="/results", tags=["evaluation"])


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "version": app.version}
