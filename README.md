# WNBA Prop Predictor

Compares a model-predicted probability of a player prop hitting against the
sportsbook's implied probability from odds you enter, then learns from
closing-line value (CLV) and actual outcomes to recalibrate itself over time.

## Quick start (Google Colab)

1. Open `WNBA_Prop_Predictor.ipynb` in Colab:
   `https://colab.research.google.com/github/ianbjorgum35-creator/WNBA/blob/claude/wnba-update/WNBA_Prop_Predictor.ipynb`
   (once this branch is merged, use `main` instead in that URL, and update
   the `BRANCH` variable in the notebook's clone cell to match).
2. Runtime > Run all.
3. Use the form: pick player team / opponent team / prop type from
   dropdowns, type in the player's name, the line, the odds, the game total
   and spread.
4. Click **Auto-Fetch Data**, then **Run Prediction**.
5. Click **Log This Prediction** to save it for the learning loop. After the
   game, come back and record the actual result (and closing odds, if you
   have them) so the model can learn from it.

No local setup required -- the notebook installs its own dependencies and
pulls this repo's `wnba_predictor` package.

## What it computes

For the player/prop you specify, it gathers (auto-fetch, with every field
manually editable/overridable):

**Player form:** Season / L20 / L10 / L5 averages and hit rates vs. your
line, minutes/game, projected minutes, usage %, starter status, rotation
role, and a teammate usage/minutes "bump" estimate (from on/off splits when
a teammate is out).

**Matchup / game environment:** opponent pace rank, opponent defensive
rank, defense-vs-position (DvP) rank, game total, spread, home/away, rest
days, back-to-back flag, head-to-head hit rate vs. this opponent, plus a
manual matchup adjustment slider and a free-text "context lean" you can
enter yourself (e.g. an injury note, a pace mismatch you noticed).

These are blended into a projected mean and variance (weighted by
recency-window blend weights that the learning loop tunes over time),
converted to a predicted probability via a Normal or Poisson model
depending on the stat, and compared against the sportsbook's implied
probability (from your entered odds) to produce an edge.

## How the learning loop works

Every prediction you log gets written to `data/bet_log.csv` with its full
input breakdown. Once you record the actual result (and closing line, if
available):

- `wnba_predictor.clv.calibration_report` gives you win rate, Brier score,
  log loss, average CLV, and a calibration curve (predicted-probability
  bucket vs. actual hit rate).
- `wnba_predictor.clv.fit_recalibration` fits a Platt-scaling logistic
  regression (raw model probability -> calibrated probability) once you
  have 20+ resolved bets, and persists it to `data/learned_weights.json`.
  The next prediction automatically runs through this calibrator.
- `wnba_predictor.clv.update_blend_weights` nudges the season/L20/L10/L5
  blend weights toward whichever recency window's hit rate has actually
  tracked outcomes best, with a bounded step size so one batch of results
  can't swing it too far.

All three require enough resolved history to be meaningful and fall back to
sane defaults (identity calibration, the config priors) until then.

## Persisting your history across Colab sessions

Colab's local disk doesn't survive a runtime recycle. In the notebook's
Drive-mount cell, set `USE_DRIVE = True` to store `bet_log.csv` and
`learned_weights.json` in your Google Drive instead, so the model keeps
learning across sessions.

## Data sources and their limits

Auto-fetch hits two unofficial, reverse-engineered public endpoints:

- `stats.wnba.com` (mirrors the well-known stats.nba.com API shape with
  `LeagueID=10`) for player game logs and team pace/defense ranks.
- ESPN's hidden site API for injuries, rosters, and schedules (rest days /
  back-to-backs).

These were not network-testable from the build sandbox (its egress policy
blocked both hosts), so **test the Auto-Fetch button first** when you open
this in Colab. They can change shape or rate-limit without notice. That's
exactly why every auto-fetched field is a plain editable box: if a fetch
fails, type the number in yourself and keep going. Defense-vs-position
(DvP) rank in particular has no single reliable free endpoint; treat
opponent defensive rank as a proxy for it unless you fill DvP in yourself.

Injury/lineup confirmation is likewise best-effort -- there's no clean
structured free feed for this, so the **Lineup Confirmed?** checkbox is the
source of truth: leave it unchecked when a rotation is uncertain, and the
model will widen its uncertainty (pulling the predicted probability toward
50%) accordingly.

## Repository layout

```
wnba_predictor/
  config.py          League/team list, prop type definitions, model defaults
  odds.py            American odds <-> implied probability, de-vig, edge, CLV, Kelly sizing
  data_sources.py     stats.wnba.com / ESPN fetchers, all wrapped to fail soft
  stats_engine.py      Rolling averages/hit rates, minutes profile, usage, teammate on/off splits
  matchup.py           Pace/defense ranks, DvP table builder, H2H hit rate, rest/b2b, game environment
  projection.py         Blend + adjustment + distribution model -> predicted probability
  clv.py               Bet logging, outcome recording, calibration report, the learning loop
  ui.py                 ipywidgets dropdown UI
WNBA_Prop_Predictor.ipynb   Self-contained Colab notebook
tests/test_pipeline.py       Synthetic-data sanity tests (no network required)
data/bet_log.csv              Seed CLV/outcome log (headers only)
```

## Running tests locally

```
pip install -r requirements.txt
pytest tests/ -v
```
