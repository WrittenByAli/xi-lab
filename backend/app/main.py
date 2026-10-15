"""XI-Lab FastAPI application — context-aware cricket team selection engine."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers import compare, live, player, recommend, results, simulate, teams

app = FastAPI(
    title="XI-Lab",
    description="Context-aware cricket team selection engine",
    version="0.3.0",
)

_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router, prefix="/recommend", tags=["selection"])
app.include_router(simulate.router, prefix="/simulate", tags=["simulation"])
app.include_router(compare.router, prefix="/compare", tags=["simulation"])
app.include_router(player.router, prefix="/player", tags=["players"])
app.include_router(live.router, prefix="/live", tags=["live"])
app.include_router(results.router, prefix="/results", tags=["evaluation"])
app.include_router(teams.router, prefix="/teams", tags=["teams"])


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "version": app.version}
