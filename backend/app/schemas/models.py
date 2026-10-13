"""Pydantic request/response models for the XI-Lab API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlayerIn(BaseModel):
    player_id: str
    name: str
    role: str = "pure_bat"
    bat_rating: float = 0.5
    bowl_rating: float = 0.5
    is_keeper: bool = False
    is_overseas: bool = False
    bat_hand: str = "R"
    bowl_type: str = "none"


class RecommendRequest(BaseModel):
    squad: list[PlayerIn] = Field(..., min_length=11)
    format: str = "t20"
    opponent_team: str | None = None
    venue: str | None = None
    max_overseas: int = 4
    n_sims: int = 5000


class PlayerSummary(BaseModel):
    player_id: str
    name: str
    role: str
    bat_rating: float
    bowl_rating: float
    is_keeper: bool
    is_overseas: bool


class RecommendResponse(BaseModel):
    xi: list[PlayerSummary]
    win_prob: float
    batting_order: list[str]
    bowling_plan: dict[str, int]
    tie_set: list[list[str]]
    swap_matrix: dict[str, dict[str, float]]


class SimulateRequest(BaseModel):
    team1: list[PlayerIn] = Field(..., min_length=11, max_length=11)
    team2: list[PlayerIn] = Field(..., min_length=11, max_length=11)
    format: str = "t20"
    n_sims: int = 10000
    venue: str | None = None


class ScoreDistribution(BaseModel):
    mean: float
    std: float
    median: float
    q10: float
    q90: float


class SimulateResponse(BaseModel):
    team1_win_prob: float
    team2_win_prob: float
    tie_prob: float
    team1_scores: ScoreDistribution
    team2_scores: ScoreDistribution


class CompareRequest(BaseModel):
    team1_a: list[PlayerIn] = Field(..., min_length=11, max_length=11)
    team1_b: list[PlayerIn] = Field(..., min_length=11, max_length=11)
    opponent: list[PlayerIn] = Field(..., min_length=11, max_length=11)
    format: str = "t20"
    n_sims: int = 10000


class CompareResponse(BaseModel):
    delta_wp: float
    ci_lower: float
    ci_upper: float
    team_a_wp: float
    team_b_wp: float


class PhaseRating(BaseModel):
    phase: str
    bat_sr: float | None = None
    bowl_econ: float | None = None
    balls: int = 0


class PlayerProfile(BaseModel):
    player_id: str
    name: str
    role: str
    bat_hand: str | None = None
    bowl_type: str | None = None
    is_keeper: bool = False
    bat_rating: float | None = None
    bowl_rating: float | None = None
    bat_sr: float | None = None
    bat_avg: float | None = None
    bowl_econ: float | None = None
    bowl_sr: float | None = None
    phase_ratings: list[PhaseRating] = []
    total_balls_bat: int = 0
    total_balls_bowl: int = 0


class PredictionRecord(BaseModel):
    match_id: str
    date: str
    team1: str
    team2: str
    predicted_wp: float
    actual_winner: str | None = None
    correct: bool | None = None


class BacktestMetrics(BaseModel):
    format: str
    test_matches: int
    brier_score: float | None = None
    log_loss: float | None = None
    accuracy: float | None = None
    kelly_roi: float | None = None
    calibration: list[dict] = []
