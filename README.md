# XI-Lab

**Context-aware cricket team selection engine for all international teams.**

Given a venue, opponent, toss result and available squad, XI-Lab recommends
the XI with the highest simulated win probability — for any team, in T20I,
ODI, and Test cricket.

> Data: [Cricsheet](https://cricsheet.org) (ODC-BY licence)

---

## Status

| Component | Status |
|-----------|--------|
| Data pipeline (Cricsheet parse/normalize) | Done |
| Player coverage (6,303 players, 98.9% T20) | Done |
| Monthly as-of snapshots (leakage-proof) | Done |
| Ability model (Empirical Bayes shrinkage) | Done |
| Ball-outcome model (LightGBM 9-class) | Done |
| Policy models (batting order + bowling plan) | Done |
| Monte Carlo simulator (vectorized, CRN) | Done |
| ILP optimizer + racing search | Done |
| Backtest evaluator (calibrated logistic) | Done |
| FastAPI backend (6 routers, 11 endpoints) | Done |
| CI/CD (GitHub Actions, Docker Compose) | Done |
| T20 format (Brier 0.236, accuracy 59.5%) | Done |
| ODI format (Brier 0.257, accuracy 55.0%) | Done |
| Test format | Planned |
| React frontend (dark theme, 5 pages) | Done |
| 112 tests passing | Done |

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

# 4. Train all models (T20 + ODI)
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

## API

```bash
uvicorn backend.app.main:app --reload
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check |
| `/recommend` | POST | Best XI from squad (ILP + racing search) |
| `/simulate` | POST | Match simulation with score distributions |
| `/compare` | POST | Paired delta-WP between two XI variants |
| `/player/{id}` | GET | Player profile with phase ratings |
| `/player/search?q=` | GET | Search players by name |
| `/live` | GET | Prediction log with results |
| `/results?format=t20` | GET | Backtest metrics (t20/odi) |
| `/results/ability` | GET | Ability model metrics |
| `/results/ball` | GET | Ball model metrics + feature importance |
| `/results/policies` | GET | Policy model metrics |
| `/results/all` | GET | All evaluation metrics |

Interactive docs at `http://localhost:8000/docs`.

---

## Frontend

React 19 + TypeScript + Vite + Tailwind CSS v4.

| Page | Path | Description |
|------|------|-------------|
| Dashboard | `/` | Health status, T20/ODI comparison, calibration preview |
| Squad Builder | `/squad` | Build squad, get optimal XI with batting order + bowling plan |
| Simulator | `/simulate` | Head-to-head simulation with WP bar + score distributions |
| Players | `/players` | Search, profile cards, phase breakdown bars |
| Results | `/results` | Calibration scatter plot, feature importance, model metrics |

All pages support T20/ODI format toggle.

---

## Model performance

### T20 (3,512 test matches)
- **Brier score**: 0.236 (baseline: 0.250)
- **Accuracy**: 59.5% (baseline: 50%)
- **Log loss**: 0.664 (baseline: 0.693)
- **Kelly ROI**: 5.7%

### ODI (469 test matches)
- **Brier score**: 0.257 (baseline: 0.250)
- **Accuracy**: 55.0%
- **Kelly ROI**: 7.3%

### Ball-outcome model
- **T20**: log_loss 1.56, accuracy 43% (9-class)
- **ODI**: log_loss 1.29, accuracy 53% (9-class)

---

## Limitations

- Cricsheet has no pitch reports. Conditions are approximated from venue history.
- Uncapped players with little data get wide uncertainty intervals.
- Domestic coverage varies by country; some uncapped players have PSL/IPL data only.
- ODI backtest is marginal (Brier 0.257 vs 0.250 baseline) due to smaller validation set.
- This is decision support, not a replacement for selectors.

---

## Citation

```
Cricsheet (https://cricsheet.org) — Open Data Commons Attribution Licence (ODC-BY)
```
