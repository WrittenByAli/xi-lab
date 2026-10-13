# XI-Lab

**Context-aware cricket team selection engine for all international teams.**

Given a venue, opponent, toss result and available squad, XI-Lab recommends
the XI with the highest simulated win probability — for any team, in T20I,
ODI, and Test cricket.

> Data: [Cricsheet](https://cricsheet.org) (ODC-BY licence)

---

## Status

| Phase | Format | Status |
|-------|--------|--------|
| 1 | T20I + all T20 leagues, all teams | 🔨 In progress |
| 2 | ODI | 📋 Planned |
| 3 | Test | 📋 Planned |

---

## Reproduce everything

```bash
# 1. Install
pip install -e ".[dev]"
pre-commit install

# 2. Download data (Cricsheet, ~2 GB)
make download

# 3. Full pipeline: parse → features → train → evaluate
make all        # runs dvc repro

# 4. Tests
make test

# 5. Serve locally
make serve
```

One command (`make all`) reproduces every result from raw data.
See `RESULTS.md` for all model metrics, calibration plots and the backtest.

---

## Architecture

```
DATA      Cricsheet JSON → Parquet (balls, innings, matches, lineups)
FEATURES  Monthly as-of snapshots, venue profiles, team Elo
MODELS    A. League-adjusted ability (empirical Bayes)
          B. Ball-outcome model (calibrated LightGBM)
          C. Batting-order + bowling-allocation policies (cond. logit)
          D. Match simulator (vectorized Monte Carlo, posterior sampling)
          E. Selection optimizer (ILP shortlist → racing search → tie set)
SERVING   FastAPI + React dashboard
MLOPS     MLflow, DVC, Docker Compose, GitHub Actions
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

---

## Live predictions

XI recommendations are published as GitHub Releases before every toss.
Each release includes the prediction JSON and the model version hash.
Results are scored after the match.

See [live/predictions](live/predictions/) and [live/results](live/results/).

---

## Results

See [RESULTS.md](RESULTS.md) for:
- Ball model log loss vs baselines, with ablations
- Calibration and reliability curves
- Pre-match backtest vs team Elo (E7)
- Lineup sensitivity test (E8)
- Selector study: where the model and selectors disagree

---

## Limitations

- Cricsheet has no pitch reports. Conditions are approximated from venue history.
- Uncapped players with little data get wide uncertainty intervals.
- Domestic coverage varies by country; some uncapped players have PSL/IPL data only.
- The simulator assumes both teams bat their full XI in the standard order;
  declarations and follow-on (Test) are modelled separately from Phase 3.
- This is decision support, not a replacement for selectors.
  A model that beats Elo on average can still be wrong in any specific match.

Full limitations: [LIMITATIONS.md](LIMITATIONS.md)

---

## Citation

```
Cricsheet (https://cricsheet.org) — Open Data Commons Attribution Licence (ODC-BY)
```
