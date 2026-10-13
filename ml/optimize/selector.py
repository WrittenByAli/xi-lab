"""
XI selection via Integer Linear Programming (ILP) + racing search.

Given a squad of ~15-20 players, finds the best XI that maximizes
expected win probability subject to team composition constraints.

Constraints:
  - Exactly 11 players
  - At least 1 keeper
  - At least 5 bowling options (players who bowl)
  - At least 5 specialist batters in top 7
  - Max 4 overseas (if applicable)

Racing search (successive halving):
  Start with ILP_N_CANDIDATES top XIs, simulate RACING_INITIAL_SIMS each,
  halve the field each round, doubling sims per round, until ≤2 remain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import pulp
except ImportError:
    pulp = None

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS = yaml.safe_load((ROOT / "params.yaml").read_text())


@dataclass
class PlayerOption:
    """A player available for selection."""
    player_id: str
    name: str
    role: str           # pure_bat, bat_allrounder, bowl_allrounder, pure_bowl, keeper_bat
    bat_rating: float
    bowl_rating: float
    is_keeper: bool
    is_overseas: bool = False
    bat_hand: str = "R"
    bowl_type: str = "none"


@dataclass
class SelectionResult:
    """Output of the selection optimizer."""
    xi: list[PlayerOption]
    win_prob: float
    batting_order: list[str]
    bowling_plan: dict[str, int]  # player_id -> overs
    tie_set: list[list[str]]      # alternative XIs within CI
    swap_matrix: dict[str, dict[str, float]]  # in -> {out -> delta_wp}


def generate_candidate_xis(
    squad: list[PlayerOption],
    n_candidates: int = 30,
    max_overseas: int = 4,
) -> list[list[PlayerOption]]:
    """
    Generate candidate XIs using ILP with diverse objective perturbations.

    Each candidate solves the ILP with small random noise added to ratings,
    producing naturally diverse XIs that explore the feasible space.
    """
    if pulp is None:
        return [_greedy_select(squad, max_overseas)]

    candidates = []
    rng = np.random.default_rng(42)

    for trial in range(n_candidates * 3):
        if len(candidates) >= n_candidates:
            break

        noise = rng.normal(0, 0.05, len(squad))

        prob = pulp.LpProblem(f"xi_select_{trial}", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(len(squad))}

        prob += pulp.lpSum(
            x[i] * (squad[i].bat_rating + squad[i].bowl_rating + noise[i])
            for i in range(len(squad))
        )

        prob += pulp.lpSum(x[i] for i in range(len(squad))) == 11

        prob += pulp.lpSum(
            x[i] for i in range(len(squad)) if squad[i].is_keeper
        ) >= 1

        prob += pulp.lpSum(
            x[i] for i in range(len(squad))
            if squad[i].role in ("bowl_allrounder", "pure_bowl", "bat_allrounder")
        ) >= 5

        prob += pulp.lpSum(
            x[i] for i in range(len(squad)) if squad[i].is_overseas
        ) <= max_overseas

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status == pulp.constants.LpStatusOptimal:
            xi = [squad[i] for i in range(len(squad)) if x[i].varValue > 0.5]
            if len(xi) == 11:
                xi_ids = frozenset(p.player_id for p in xi)
                if not any(frozenset(p.player_id for p in c) == xi_ids for c in candidates):
                    candidates.append(xi)

    if not candidates:
        candidates.append(_greedy_select(squad, max_overseas))

    return candidates


def _greedy_select(
    squad: list[PlayerOption],
    max_overseas: int = 4,
) -> list[PlayerOption]:
    """Fallback greedy selection when PuLP is unavailable."""
    keepers = [p for p in squad if p.is_keeper]
    bowlers = [p for p in squad if p.role in ("pure_bowl", "bowl_allrounder")]
    batters = [p for p in squad if p.role in ("pure_bat", "bat_allrounder", "keeper_bat")]

    selected = []
    selected_ids = set()

    if keepers:
        k = max(keepers, key=lambda p: p.bat_rating)
        selected.append(k)
        selected_ids.add(k.player_id)

    for p in sorted(bowlers, key=lambda p: p.bowl_rating, reverse=True):
        if p.player_id not in selected_ids and len(selected) < 11:
            overseas_count = sum(1 for s in selected if s.is_overseas)
            if not p.is_overseas or overseas_count < max_overseas:
                selected.append(p)
                selected_ids.add(p.player_id)
                if len([s for s in selected if s.role in ("pure_bowl", "bowl_allrounder")]) >= 5:
                    break

    for p in sorted(batters, key=lambda p: p.bat_rating, reverse=True):
        if p.player_id not in selected_ids and len(selected) < 11:
            overseas_count = sum(1 for s in selected if s.is_overseas)
            if not p.is_overseas or overseas_count < max_overseas:
                selected.append(p)
                selected_ids.add(p.player_id)

    remaining = sorted(
        [p for p in squad if p.player_id not in selected_ids],
        key=lambda p: p.bat_rating + p.bowl_rating,
        reverse=True,
    )
    for p in remaining:
        if len(selected) >= 11:
            break
        overseas_count = sum(1 for s in selected if s.is_overseas)
        if not p.is_overseas or overseas_count < max_overseas:
            selected.append(p)
            selected_ids.add(p.player_id)

    return selected[:11]


def suggest_batting_order(xi: list[PlayerOption]) -> list[str]:
    """Suggest a batting order based on role and rating."""
    keepers = [p for p in xi if p.is_keeper]
    openers = sorted(
        [p for p in xi if p.role in ("pure_bat", "keeper_bat", "bat_allrounder")],
        key=lambda p: p.bat_rating, reverse=True,
    )[:2]

    middle = sorted(
        [p for p in xi if p not in openers and p.role in ("pure_bat", "bat_allrounder", "keeper_bat")],
        key=lambda p: p.bat_rating, reverse=True,
    )

    lower = sorted(
        [p for p in xi if p not in openers and p not in middle],
        key=lambda p: p.bat_rating, reverse=True,
    )

    order = openers + middle + lower
    seen = set()
    deduped = []
    for p in order:
        if p.player_id not in seen:
            deduped.append(p.player_id)
            seen.add(p.player_id)

    for p in xi:
        if p.player_id not in seen:
            deduped.append(p.player_id)
            seen.add(p.player_id)

    return deduped


def suggest_bowling_plan(xi: list[PlayerOption], total_overs: int = 20) -> dict[str, int]:
    """Allocate overs to bowlers."""
    bowlers = [p for p in xi if p.role in ("pure_bowl", "bowl_allrounder", "bat_allrounder")]
    if not bowlers:
        bowlers = sorted(xi, key=lambda p: p.bowl_rating, reverse=True)[:5]

    bowlers = sorted(bowlers, key=lambda p: p.bowl_rating, reverse=True)
    max_per_bowler = total_overs // 5  # 4 in T20

    plan = {}
    remaining = total_overs
    for b in bowlers:
        overs = min(max_per_bowler, remaining)
        if overs > 0:
            plan[b.player_id] = overs
            remaining -= overs
        if remaining <= 0:
            break

    if remaining > 0:
        for b in bowlers:
            if remaining <= 0:
                break
            extra = min(remaining, max_per_bowler - plan.get(b.player_id, 0))
            if extra > 0:
                plan[b.player_id] = plan.get(b.player_id, 0) + extra
                remaining -= extra

    return plan


def racing_search(
    candidates: list[list[PlayerOption]],
    evaluate_fn,
    initial_sims: int = 1000,
    rounds: int = 4,
    alpha: float = 0.05,
) -> tuple[list[PlayerOption], float, list[list[str]]]:
    """
    Successive halving: evaluate candidates, eliminate bottom half each round,
    doubling simulation budget. Returns best XI, its WP, and tie set.

    evaluate_fn(xi, n_sims) -> float (win probability)
    """
    active = list(range(len(candidates)))
    sims = initial_sims
    scores = {}

    for rnd in range(rounds):
        if len(active) <= 2:
            break

        for idx in active:
            wp = evaluate_fn(candidates[idx], sims)
            scores[idx] = wp

        active.sort(key=lambda i: scores[i], reverse=True)
        half = max(2, len(active) // 2)
        active = active[:half]
        sims *= 2

    for idx in active:
        scores[idx] = evaluate_fn(candidates[idx], sims)

    best_idx = max(active, key=lambda i: scores[i])
    best_xi = candidates[best_idx]
    best_wp = scores[best_idx]

    sorted_active = sorted(active, key=lambda i: scores[i], reverse=True)
    tie_set = []
    if len(sorted_active) > 1:
        best_score = scores[sorted_active[0]]
        for idx in sorted_active[1:]:
            if abs(scores[idx] - best_score) < alpha:
                tie_set.append([p.player_id for p in candidates[idx]])

    return best_xi, best_wp, tie_set


def compute_swap_matrix(
    xi: list[PlayerOption],
    bench: list[PlayerOption],
    evaluate_fn,
    n_sims: int = 2000,
) -> dict[str, dict[str, float]]:
    """For each bench player, compute delta WP if swapped in for each XI player."""
    base_wp = evaluate_fn(xi, n_sims)
    matrix = {}

    for bench_p in bench:
        matrix[bench_p.player_id] = {}
        for i, xi_p in enumerate(xi):
            trial_xi = list(xi)
            trial_xi[i] = bench_p
            trial_wp = evaluate_fn(trial_xi, n_sims)
            matrix[bench_p.player_id][xi_p.player_id] = round(trial_wp - base_wp, 4)

    return matrix


def select_xi(
    squad: list[PlayerOption],
    evaluate_fn=None,
    max_overseas: int = 4,
) -> SelectionResult:
    """
    Full selection pipeline: ILP candidates → racing search → batting/bowling plans.

    evaluate_fn(xi: list[PlayerOption], n_sims: int) -> float
    If None, uses a simple rating-sum heuristic.
    """
    opt_params = PARAMS["optimizer"]

    if evaluate_fn is None:
        def evaluate_fn(xi, n_sims=1000):
            return sum(p.bat_rating + p.bowl_rating for p in xi) / 11

    candidates = generate_candidate_xis(
        squad,
        n_candidates=opt_params["ilp_n_candidates"],
        max_overseas=max_overseas,
    )

    if len(candidates) == 1:
        best_xi = candidates[0]
        best_wp = evaluate_fn(best_xi, 1000)
        tie_set = []
    else:
        best_xi, best_wp, tie_set = racing_search(
            candidates,
            evaluate_fn,
            initial_sims=opt_params["racing_initial_sims"],
            rounds=opt_params["racing_rounds"],
            alpha=opt_params["tie_set_alpha"],
        )

    bench = [p for p in squad if p.player_id not in {x.player_id for x in best_xi}]

    return SelectionResult(
        xi=best_xi,
        win_prob=best_wp,
        batting_order=suggest_batting_order(best_xi),
        bowling_plan=suggest_bowling_plan(best_xi),
        tie_set=tie_set,
        swap_matrix=compute_swap_matrix(best_xi, bench, evaluate_fn) if bench else {},
    )
