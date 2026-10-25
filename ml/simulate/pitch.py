"""
Manual pitch-condition overrides for the ball-outcome probability vector.

Cricsheet has no pitch reports (see README limitations), so the primary,
honest conditioning on ground character already happens through the real
ball-outcome model's `venue_id` feature in `cards.py` — a venue's own
scoring/wicket history shapes the prediction automatically. The presets here
are an *optional* explicit override for when a user wants to simulate
conditions that diverge from (or substitute for, e.g. a venue with little
history) that learned venue effect — "it's underprepared and turning more
than usual today", or a hypothetical scenario.

Each preset nudges the 9-class outcome vector
[dot, 1, 2, 3, 4, 6, wicket, wide, noball] by small, fixed multipliers,
re-normalized afterwards. "auto" (the default everywhere) and "balanced"
are no-ops.
"""

from __future__ import annotations

import numpy as np

OUTCOME_CLASSES = ["dot", "1", "2", "3", "4", "6", "wicket", "wide", "noball"]
_IDX = {name: i for i, name in enumerate(OUTCOME_CLASSES)}

# bowl_type vocabulary, from data/reference/player_meta.csv.
SPIN_TYPES = frozenset({"off", "left_orth", "leg", "left_wrist"})
PACE_TYPES = frozenset({"pace"})

PITCH_CONDITIONS = [
    "auto",
    "balanced",
    "batting_friendly",
    "bowling_friendly",
    "spin_friendly",
    "seam_friendly",
]


def _renormalize(probs: np.ndarray) -> np.ndarray:
    probs = np.maximum(probs, 1e-6)
    return probs / probs.sum()


def apply_pitch_adjustment(
    probs: np.ndarray,
    pitch_condition: str | None,
    bowl_type: str | None = None,
) -> np.ndarray:
    """Apply a named pitch-condition preset to a 9-class outcome vector.

    `bowl_type` is the specific bowler's type when adjusting a BowlingCard
    (the type-conditional presets then apply their full effect only to the
    matching style). Pass `None` when adjusting a BattingCard against a
    generic/blended attack — type-conditional presets then fall back to a
    milder, type-agnostic version, since the specific bowler isn't fixed.
    Never mutates the input array.
    """
    if pitch_condition in (None, "auto", "balanced"):
        return probs

    p = np.asarray(probs, dtype=float).copy()
    dot, four, six, wkt = _IDX["dot"], _IDX["4"], _IDX["6"], _IDX["wicket"]

    if pitch_condition == "batting_friendly":
        p[dot] *= 0.90
        p[four] *= 1.15
        p[six] *= 1.15
        p[wkt] *= 0.85
    elif pitch_condition == "bowling_friendly":
        p[dot] *= 1.10
        p[four] *= 0.85
        p[six] *= 0.85
        p[wkt] *= 1.20
    elif pitch_condition in ("spin_friendly", "seam_friendly"):
        favored = SPIN_TYPES if pitch_condition == "spin_friendly" else PACE_TYPES
        other = PACE_TYPES if pitch_condition == "spin_friendly" else SPIN_TYPES

        if bowl_type in favored:
            p[dot] *= 1.12
            p[wkt] *= 1.25
            p[four] *= 0.88
            p[six] *= 0.80
        elif bowl_type in other:
            p[four] *= 1.05  # the "wrong" bowling type is comparatively easier to score off
        else:
            # Generic/blended attack (e.g. a BattingCard with no single fixed
            # bowler): apply a milder, type-agnostic version of the same
            # direction, since on average the pitch still favours bowlers.
            p[dot] *= 1.05
            p[wkt] *= 1.10
            p[four] *= 0.95
    else:
        return probs  # unknown preset name — no-op rather than raising

    return _renormalize(p)
