# XI-Lab

**Context-aware cricket team selection engine, for men's and women's cricket.**

Given a gender, ground, opponent, toss decision, pitch condition, and
available squad, XI-Lab recommends the XI with the highest simulated win
probability — for any team, in T20I, ODI, and Test cricket. Squads can be
built from a team's current roster or its all-time player pool.

> Data: [Cricsheet](https://cricsheet.org) (ODC-BY licence)

---

## Status

| Component | Status |
|-----------|--------|
| Data pipeline (Cricsheet parse/normalize, men's + women's) | Done |
| Player coverage (6,303+ players, 98.9% T20) | Done |
| Monthly as-of snapshots (leakage-proof, per gender) | Done |
| Ability model (Empirical Bayes shrinkage, per gender) | Done |
| Ball-outcome model (LightGBM 9-class, venue/toss-aware, per gender) | Done |
| Policy models (batting order + bowling plan, per gender) | Done |
| Monte Carlo simulator (CRN, ground/pitch/toss-aware, fully vectorized) | Done |
| ILP optimizer + racing search (opponent-aware when selected) | Done |
| Backtest evaluator (calibrated logistic, per gender) | Done |
| FastAPI backend (7 routers, 16 endpoints) | Done |
| CI/CD (GitHub Actions, Docker Compose) | Done |
| T20 format — men's & women's | Done |
| ODI format — men's & women's | Done |
| Test format — men's & women's (ability + ball + policy; backtest men's only, see Limitations) | Done |
| React frontend (dark theme, 6 pages, men's/women's toggle) | Done |
| Live prediction pipeline (`publish.py` → GitHub Release → `record_result.py` → `/live`) | Done |
| Tests passing | Done |

---

## Quick start

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Download data (Cricsheet, ~2 GB)
make download

# 3. Parse and build features
make parse
make features

# 4. Train all models (T20 + ODI + Test)
make train

# 5. Backtest
make backtest

# 6. Tests
make test

# 7. Serve API + frontend
make serve          # backend on :8000
make serve-frontend # frontend on :3000
```

Or with Docker:
```bash
docker compose up
# frontend on :3000, API on :8000
```

---

## Architecture

```
DATA      Cricsheet JSON → Parquet (balls, innings, matches, lineups)
FEATURES  Monthly as-of snapshots (leakage-proof, 730-day decay)
MODELS    A. Ability ratings (empirical Bayes shrinkage, per-format)
          B. Ball-outcome model (LightGBM 9-class multiclass)
          C. Batting-order + bowling-allocation policies
          D. Match simulator (vectorized Monte Carlo, CRN)
          E. Selection optimizer (ILP shortlist → racing search → tie set)
EVAL      Calibrated logistic regression on z-scored team rating diffs
SERVING   FastAPI backend + React/TypeScript frontend
DEPLOY    Docker Compose, GitHub Actions CI
```

---

## Key design decisions

- **No player IDs as model features.** Ratings generalize to debutants.
- **Leakage-proof.** Every feature uses only data before the match date.
  Automated truncation and poison tests enforce this in CI.
- **Uncertainty everywhere.** Posterior sampling propagates rating uncertainty
  through simulations. The UI shows ranges, not single numbers.
- **Common random numbers.** All candidate XIs face identical random draws,
  so comparisons are paired and noise is minimal.
- **Tie sets.** The output is the best XI *and* every XI statistically
  indistinguishable from it — with a core (always selected) and flexible players.
- **Calibrated backtest.** Logistic regression fitted on validation set with
  z-score normalization, evaluated on held-out test set.

---

## Performance

Two changes account for most of the latency in this project, both fixed
by actually vectorizing code that already *claimed* to be vectorized:

- **Match simulator**: `ml/simulate/engine.py`'s docstring said "fully
  vectorized... no Python per-ball loop," but the actual implementation
  looped over every simulation at every delivery (e.g. an ODI's ~340
  deliveries × 10,000 sims = 3.4M scalar Python iterations per match
  simulated). Rewritten to vectorize across simulations with numpy array
  ops instead — verified bit-identical output against the old code for
  the same random seed (same RNG draws, same math, just not drawn one
  simulation at a time), then benchmarked: **10,000-sim T20 match: 25.0s
  → 0.46s (54x)**. This is the simulator every `/recommend`, `/simulate`,
  `/compare`, backtest, and live-publish call goes through.
- **Backtest**: `_build_dataset` rescanned the *entire* lineups table
  (222,680 rows for t20/male) and the entire ratings table from scratch
  for every match, twice over (once per team) — O(matches × lineups)
  instead of O(matches). Rewritten to index both once (lineups by
  `(match_id, team)`, ratings per player as sorted numpy arrays with a
  binary-search as-of lookup) and reuse the index across both the
  validation and test windows. **Full t20/male backtest: ~85s → ~4s**,
  and the live-publish pipeline's final high-fidelity win-probability
  evaluation (previously the one case that could exceed 120s and need to
  run in the background) now finishes in well under 20s.

Typical end-to-end `/recommend` latency against the live API (real squad,
real opponent, context-aware simulation), measured directly rather than
estimated: ~7-9s for T20/ODI, ~13-18s for Test (its 90-over innings make
it the slowest of the three formats).

---

## API

```bash
uvicorn backend.app.main:app --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/recommend` | POST | Best XI from squad (ILP + racing search); response includes `warnings` for degenerate squads (e.g. no recognized keeper) |
| `/simulate` | POST | Match simulation with score distributions |
| `/compare` | POST | Paired delta-WP between two XI variants (simulates a full match for each — see Limitations for a fixed bug where this used to compare projected 1st-innings scores instead) |
| `/player/{id}` | GET | Player profile with phase ratings |
| `/player/search?q=` | GET | Search players by name |
| `/live` | GET | Prediction log with results |
| `/results?format=t20` | GET | Backtest metrics (t20/odi/test) |
| `/results/ability` | GET | Ability model metrics |
| `/results/ball` | GET | Ball model metrics + feature importance |
| `/results/policies` | GET | Policy model metrics |
| `/results/all` | GET | All evaluation metrics |

Interactive docs at `http://localhost:8000/docs`.

---

## Live predictions

A scheduled GitHub Action (`publish_prediction.yml`) runs
`ml/optimize/publish.py` before a real match: given a squad, opponent,
venue, and toss/pitch context, it runs the same ILP + racing search +
context-aware simulator the `/recommend` endpoint uses, then commits the
XI, win probability, batting order, bowling plan, and swap matrix to
`live/predictions/<team>_vs_<opponent>_<date>.json` and tags a GitHub
Release with the model hash — a verifiable, timestamped, pre-toss record.

After the match, `ml/optimize/record_result.py` takes that prediction file
plus the actual winner and writes two things: a per-match audit file in
`live/results/` (the result-side counterpart to the prediction file), and
an entry in `data/processed/predictions/{format}_predictions.json`, which
is what `/live` and `/live/stats` actually serve — tracked accuracy over
every prediction made this way, not just the matches the backtest already
covers.

```bash
python ml/optimize/record_result.py \
    --prediction live/predictions/India_vs_Australia_2024-03-15.json \
    --winner India
```

---

## Frontend

React 19 + TypeScript + Vite + Tailwind CSS v4.

| Page | Path | Description |
|------|------|-------------|
| Dashboard | `/` | Health status, T20/ODI comparison, calibration preview |
| Squad Builder | `/squad` | Build squad, get optimal XI with batting order + bowling plan |
| Simulator | `/simulate` | Head-to-head simulation with WP bar + score distributions |
| Players | `/players` | Search, profile cards, phase breakdown bars |
| Compare | `/compare` | Paired delta-WP between two XI variants against the same opponent |
| Results | `/results` | Calibration scatter plot, feature importance, model metrics |

All pages support T20/ODI/Test format toggle, and men's/women's.

---

## Model performance

Backtest: calibrated logistic fit on validation-period matches (2023),
evaluated on held-out test-period matches (2024 onwards). Numbers below are
measured directly off the current data snapshot — this table is
regenerated by `make backtest`, not hand-maintained, so it won't drift out
of sync with the data again.

### Match-outcome backtest

| Format | Gender | Test matches | Brier (baseline 0.250) | Accuracy | Kelly ROI |
|--------|--------|--------------|-------------------------|----------|-----------|
| T20  | male   | 2,939 | 0.245 | 55.3% | 1.5% |
| T20  | female | 1,193 | 0.225 | 61.8% | 10.8% |
| ODI  | male   | 271   | 0.259 | 46.9% | -1.4% |
| ODI  | female | 198   | 0.228 | 62.6% | 8.3% |
| Test | male   | 109   | 0.292 | 52.3% | 1.3% |
| Test | female | —     | insufficient data (24 women's Tests exist, ever) | — | — |

ODI men's is the one format currently **worse than baseline**, and it's
traceable to a specific cause, not generic noise. The 2023 validation
window is 56% ICC World Cup / World Cup Qualifier matches — mostly
full-member teams. The 2024+ test window is 42% **ICC Men's Cricket World
Cup League 2** (Nepal, Oman, Namibia, UAE, Canada, Netherlands, Scotland,
USA), which barely registers in validation (10%) and has visibly worse
player-rating coverage (80.6% of lineup players rated, vs 94.3%
elsewhere) — fewer recorded matches for these players means noisier
ability ratings feeding the win-probability fit. Splitting the test set
confirms it: League 2 matches score 44.3% accuracy / 0.265 Brier, while
every other test match scores 48.7% accuracy / 0.255 Brier — both close
to baseline, neither anywhere near as alarming as the blended 0.259
number alone suggests. The fit itself is calibrated on whichever
population the validation window happens to contain, and 2024+ simply
shifted which population dominates the test window. Filtering those
matches out of the backtest would flatter the headline number, but this
project's whole design point is that ratings should generalize to
low-data players — so that's reported here instead of quietly excluded.

### Ball-outcome model (9-class, per legal delivery)

| Format | Gender | Log loss | Accuracy |
|--------|--------|----------|----------|
| T20  | male   | 1.580 | 41.6% |
| T20  | female | 1.483 | 44.8% |
| ODI  | male   | 1.323 | 50.9% |
| ODI  | female | 1.265 | 53.1% |
| Test | male   | 1.013 | 69.8% |
| Test | female | 0.996 | 71.2% |

Test cricket's higher accuracy reflects its outcome mix (far more dot
balls), not an easier prediction task — argmax collapses toward "dot" for
every format here once a class's base rate dominates; the simulator
samples from the full probability vector, not the argmax, which is what
these models are trained and used for.

---

## Limitations

- Cricsheet has no pitch reports. Conditions are approximated from venue history.
- Uncapped players with little data get wide uncertainty intervals.
- Domestic coverage varies by country; some uncapped players have PSL/IPL data only.
- Cricsheet has no 4th-innings target for Test matches, so `rrr_before` is
  always null there; the ball model handles this as a genuinely-missing
  feature rather than needing a separate model shape per format.
- **`is_international` was wrong for most of the dataset, not just
  women's cricket.** It was originally decided by matching the
  competition name against `league_map.csv`'s ~17 well-known men's
  leagues/tournaments. That approach was tried once already for the long
  tail of women's bilateral tour names (a "contains 'women'" fallback)
  and fixed that gender — but never re-checked for men's cricket, where
  it was just as broken: ordinary tour names ("Australia tour of India")
  and even major ICC events ("ICC Champions Trophy", "ICC World Cup")
  never matched anything, so men's ODI measured **5.0% international**
  and men's Test measured **0.0%** — every Test is international by
  definition. Replaced entirely with team-identity matching
  (`ml.ingest.normalize.is_international_match`): a match is
  international iff both sides are national teams, which needs no
  competition-name enumeration at all. The ~259 team names appearing
  anywhere in this dataset were individually classified once (the
  reasoning, including the deliberately-excluded edge cases — "Barbados"
  plays only as a West Indian regional/franchise side here, honorary
  teams like "Asia XI" are excluded — is in `COUNTRY_TEAMS`'s comment).
  Resulting rates: t20/male 39.3%, t20/female 73.0%, odi/male 99.7%,
  odi/female 100.0%, test/male 99.9%, test/female 100.0% — now coherent
  with how much of each format is actually played domestically. Ball
  models retrained on the corrected data; the practical effect on
  accuracy was negligible either way (the feature already had enough
  other signal to lean on), but the stored data was simply wrong, which
  would have bitten anything built on it later.
- **`_compute_features`'s `bowl_top3` used `nsmallest`, picking each
  team's 3 *worst* bowlers** (most negative bowl_rating) instead of its 3
  best (least negative — see the sign-convention entry below). Logistic
  regression partly self-corrects a consistently-flipped feature by
  learning the opposite-sign weight, so this wasn't silently producing
  nonsense, but it was computing a different, mislabeled statistic.
  Fixed to `nlargest`, matching `bat_top3`. Regression-tested in
  `tests/unit/test_backtest.py`.
- ODI men's backtest is currently below baseline — concentrated in
  associate-nation fixtures with lower rating coverage that are
  over-represented in the test window relative to validation (see Model
  performance for the breakdown). Not a calibration bug introduced this
  round: the unmodified original fit reproduces the identical number on
  the current data (verified by literally reverting the regularization
  change and rerunning).
- Women's Test backtest is unavailable, not just weak: Cricsheet has 24
  women's Tests ever recorded, which leaves 3 matches in the validation
  window — too few to fit, let alone calibrate, a 4-feature logistic model.
  Ability, ball-outcome, and policy models still train fine for women's
  Test; it's specifically the backtest's date-split validation step that
  has nothing to work with.
- The live-prediction pipeline (`publish.py` + `record_result.py`) has one
  real, verified run end-to-end as of this writing (South Africa vs India,
  2026-02-22 T20 World Cup) — real 15-man squads, real opponent/venue/toss
  context, real result recorded against it. It is not yet a track record;
  it's a demonstration that the wiring between "publish a prediction" and
  "the API that reports accuracy" actually connects, which it previously
  did not (the two sides wrote to and read from different files). The
  model called this one wrong (gave South Africa 37.4%; they won), which
  is reported as-is rather than swapped for a more flattering example.
- **`/compare` wasn't comparing win probability at all.** It simulated
  one innings per variant (`simulate_innings`, not `simulate_match`) and
  returned the *mean projected score*, with no opponent chase ever
  simulated — `team_a_wp`/`team_b_wp` were literally `mean_score / 200`,
  an arbitrary normalization that happened to land in a plausible-looking
  range, and `delta_wp` was a raw run difference (e.g. `1.49`, nothing
  close to a valid probability delta). Found by end-to-end testing real
  squads rather than unit-level checks, which is exactly the kind of bug
  that passes a unit test asserting the response *has* the right fields
  without checking the values mean what they claim to. Fixed to simulate
  a full match per variant against the same opponent, with both variants
  sharing a `crn_seed` so the comparison is properly paired (the opponent
  faces identical draws in both runs) — `delta_wp`'s confidence interval
  now comes from the actual per-simulation paired win/loss difference,
  not an ad hoc formula on two independent run totals.
- **`/recommend` could silently return an XI with no wicketkeeper.** The
  ILP's "≥1 keeper" constraint is infeasible when the squad has none to
  begin with (confirmed with a real case: Mohammad Rizwan, Pakistan's
  specialist keeper, hasn't played a T20I since Dec 2024, so he's
  correctly outside the "last 12 months" squad pool — not a bug, just a
  squad that happens to have no recent keeper) — PuLP reports every
  candidate infeasible, and the greedy fallback that kicks in next
  doesn't add a keeper either, since there's: `if keepers: ...` and the
  list is empty. The result was a plausible-looking but unfieldable XI
  with no indication anything was wrong. `/recommend` now returns a
  `warnings` field and fills it when the input squad has no flagged
  keeper, instead of silently degrading.
- **The XI-selection objective had `bowl_rating`'s sign backwards.**
  `bowl_rating` is `-(shrunk economy + shrunk strike rate)/2` — *less*
  negative is better (elite bowlers: -8 to -15; non-bowlers shrunk toward
  their role group's poor-emergency-over prior: -50 to -110+). The
  normalization used everywhere ratings feed a probability
  (`ml.simulate.cards._norm_bowl`) had this backwards — it ranked a
  specialist batter who's never bowled *above* Bumrah. Confirmed via the
  real bowl_rating distribution in
  `data/processed/models/ability/male/ratings.parquet` (percentiles, not
  guesswork) and fixed in both `cards.py` and the ILP candidate-generation
  objective in `ml/optimize/selector.py` (which also summed raw
  `bat_rating + bowl_rating` — scales that aren't comparable, so the
  objective was dominated by batting regardless of the sign bug).
  Concretely, before either fix, a real India ODI XI recommendation
  against Australia dropped Bumrah, Shami *and* Rohit Sharma entirely.
  Every "unknown rating" default that relied on the old sign (frontend
  and backend) was updated to match (`-50.0`, not `0.5`/`-0.5`).
  Regression-tested in `tests/unit/test_selector.py`.
- **Remaining, deliberately not fixed:** even with both of the above
  corrected, the objective still sums normalized bat + bowl in equal
  weight, which structurally favors all-rounders over pure specialists —
  a specialist's weak secondary skill drags their sum below an
  all-rounder's two merely-good skills, even when the specialist is
  elite at the one thing the XI needs them for. In the same India-vs-
  Australia test case, Bumrah and Shami were still excluded after the
  sign/scale fix. The existing "≥5 bowling options" constraint doesn't
  prevent this — it's satisfied by any mix of pure bowlers and
  occasional-bowling all-rounders, so the objective is free to swap out
  genuine strike bowlers entirely. A real fix needs role-aware weighting
  (e.g. weight bowling heavily for pure_bowl, lightly for pure_bat) or a
  minimum-specialist-bowlers constraint, both of which are judgment calls
  on actual weights that deserve their own validation pass, not a
  same-session follow-on to a sign-convention bug fix.
- **Squad Builder, Simulator and Compare's player search added every
  result with a flat, neutral rating**, discarding whatever real ability
  data that player actually had — `/player/search` is a lightweight bio
  lookup (name/hand/role only, no rating), and the three pages never
  followed it up with a profile fetch before adding the player to a
  squad. Every player added this way looked statistically identical to
  the optimizer, which made any squad built via search (rather than via
  "load a team's roster", which already worked correctly) effectively
  meaningless to recommend from. Fixed by fetching `/player/{id}` right
  after a search result is picked and replacing the placeholder once it
  resolves — falls back to the neutral placeholder only for a genuine
  debutant with no profile at all.
- Minor dead-code cleanup this round: two empty scaffold directories
  (`ml/models/roles/`, `ml/models/style_clf/`), two placeholder-opponent
  helpers in `ml/simulate/cards.py` that every real caller had already
  stopped using in favor of computing the same mean inline, an unused
  `win_probability()` wrapper in `ml/simulate/engine.py`, and an unused
  `load_backtest_metrics()` in `backend/app/services/selection.py`.
- This is decision support, not a replacement for selectors.

---

## Citation

```
Cricsheet (https://cricsheet.org) — Open Data Commons Attribution Licence (ODC-BY)
```
