"""Tests for the FastAPI backend endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "version" in data


def _make_player(pid: str, role: str = "pure_bat", **kwargs):
    return {
        "player_id": pid,
        "name": pid,
        "role": role,
        "bat_rating": kwargs.get("bat_rating", 0.5),
        "bowl_rating": kwargs.get("bowl_rating", 0.5),
        "is_keeper": kwargs.get("is_keeper", False),
        "is_overseas": kwargs.get("is_overseas", False),
        "bat_hand": "R",
        "bowl_type": "none",
    }


def _make_squad(n: int = 15):
    players = []
    players.append(_make_player("keeper1", "keeper_bat", is_keeper=True, bat_rating=0.6))
    for i in range(4):
        players.append(_make_player(f"bat{i}", "pure_bat", bat_rating=0.5 + i * 0.05))
    for i in range(5):
        players.append(_make_player(f"bowl{i}", "pure_bowl", bowl_rating=0.5 + i * 0.05))
    players.append(_make_player("alr1", "bat_allrounder", bat_rating=0.5, bowl_rating=0.4))
    for i in range(n - 11):
        players.append(_make_player(f"bench{i}", "pure_bat", bat_rating=0.3))
    return players


def test_recommend_basic():
    squad = _make_squad(15)
    r = client.post("/recommend", json={"squad": squad, "format": "t20"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["xi"]) == 11
    assert "batting_order" in data
    assert "bowling_plan" in data
    assert 0 <= data["win_prob"] <= 10


def test_recommend_too_few_players():
    squad = _make_squad(15)[:5]
    r = client.post("/recommend", json={"squad": squad, "format": "t20"})
    assert r.status_code == 422


def test_simulate_basic():
    squad = _make_squad(15)
    team1 = squad[:11]
    team2 = squad[:11]
    r = client.post(
        "/simulate",
        json={
            "team1": team1,
            "team2": team2,
            "format": "t20",
            "n_sims": 100,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert abs(data["team1_win_prob"] + data["team2_win_prob"] + data["tie_prob"] - 1.0) < 0.01
    assert "team1_scores" in data


def test_player_not_found():
    r = client.get("/player/NONEXISTENT_PLAYER_XYZ")
    assert r.status_code == 404


def test_results():
    r = client.get("/results")
    assert r.status_code == 200
    data = r.json()
    assert "format" in data
    assert isinstance(data["test_matches"], int)


def test_live_empty():
    r = client.get("/live")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_teams_list():
    r = client.get("/teams?format=t20")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    if data:
        assert "team" in data[0]
        assert "recent_matches" in data[0]


def test_teams_squad():
    teams = client.get("/teams?format=t20").json()
    if teams:
        team_name = teams[0]["team"]
        r = client.get(f"/teams/{team_name}/squad?format=t20")
        assert r.status_code == 200
        data = r.json()
        assert data["team"] == team_name
        assert len(data["players"]) > 0
        assert "bat_rating" in data["players"][0]


def test_compare_basic():
    squad = _make_squad(15)
    team_a = squad[:11]
    team_b = squad[1:12]
    opponent = squad[:11]
    r = client.post(
        "/compare",
        json={
            "team1_a": team_a,
            "team1_b": team_b,
            "opponent": opponent,
            "format": "t20",
            "n_sims": 100,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "delta_wp" in data
    assert "ci_lower" in data
    assert "ci_upper" in data
    assert "team_a_wp" in data
    assert "team_b_wp" in data
    assert data["ci_lower"] <= data["delta_wp"] <= data["ci_upper"]
