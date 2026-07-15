# WNBA Prop Predictor

Compares a model-predicted probability of a player prop hitting against the
sportsbook's implied probability from odds you enter, then learns from
closing-line value (CLV) and actual outcomes to recalibrate itself over time.

## Quick start (Google Colab)

1. Open `WNBA_Prop_Predictor.ipynb` in Colab:
   `https://colab.research.google.com/github/ianbjorgum35-creator/WNBA/blob/claude/wnba-update/WNBA_Prop_Predictor.ipynb`
   (once this branch is merged, use `main` instead in that URL, and update
   the `BRANCH` variable in the notebook's clone cell to match).
2. Runtime > Run all. The "Check that the live data sources are actually
   reachable" cell runs a diagnostic against every data source and prints
   which ones are working -- if `stats.wnba.com` and/or ESPN are blocked in
   your environment, you'll see it here before you even open the form.
3. Use the form: pick player team / opponent team / prop type from
   dropdowns, type in the player's name, the line, the odds, the game total
   and spread. Those (plus prop type/line/odds) are the only fields meant to
   be manual -- everything else (stats, ranks, home/away, rest days) is
   supposed to auto-fetch.
4. Click **Auto-Fetch Data**, then **Run Prediction**. If some fields don't
   populate, click **Run Diagnostics** to see exactly which data source
   failed and why, then fill those specific fields in by hand.
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

**Confirmed (via a live diagnostics run from Colab): `stats.wnba.com` is
blocked outright** -- every call to it times out after several seconds,
consistent with it blackholing requests from cloud/datacenter IPs. This
isn't a bug to chase further; the app is built to route around it
entirely via ESPN:

- ESPN's hidden site API is confirmed reachable and is the primary source
  in practice: team/player lookups, rosters, schedules (home/away, rest
  days, back-to-back -- derived automatically, not asked for manually),
  and **player game logs** (confirmed working end-to-end from a live Colab
  run).
- Team pace/defense ranks try ESPN's standings endpoint first (one
  request); if that endpoint's stat set doesn't include scoring stats
  (confirmed to happen -- standings often only carries W-L-PCT-type
  columns), it falls through to querying each team's own `/statistics`
  endpoint instead (slower, one request per team, but independent of what
  standings exposes).
- `stats.wnba.com` (mirrors the stats.nba.com API shape with `LeagueID=10`)
  is tried first for player game logs and team pace/defense ranks, on the
  chance it's reachable from your environment, but is not required --
  everything falls through to the ESPN path above when it isn't.

Since neither host was network-testable from the build sandbox itself,
**run the diagnostics cell/button** any time something isn't populating --
it hits every endpoint independently (including the actual gamelog parser
and both team-ranks fallback tiers) and reports exactly which one is
failing and why (timeout vs. HTTP error vs. an unexpected response shape).
Watch the `espn_*` lines specifically, since those are the paths that
actually matter now. That's also why every auto-fetched field stays a
plain editable box regardless: if a fetch ever fails, type the number in
yourself and keep going rather than being blocked.

Remaining known gap:
- **Defense-vs-position (DvP) rank** has no reliable free endpoint at all,
  auto-fetch or otherwise; treat opponent defensive rank as a proxy for it,
  or fill it in yourself.
- The ESPN-derived pace rank is a **proxy** (combined scoring per game),
  not true possession-based pace like `stats.wnba.com` would give you --
  directionally useful, not numerically identical.

Injury/lineup confirmation is likewise best-effort -- there's no clean
structured free feed for this, so the **Lineup Confirmed?** checkbox is the
source of truth: leave it unchecked when a rotation is uncertain, and the
model will widen its uncertainty (pulling the predicted probability toward
50%) accordingly.

## Repository layout

```
wnba_predictor/
  config.py          League/team list (15 teams, 2026 season), prop type definitions, model defaults
  odds.py            American odds <-> implied probability, de-vig, edge, CLV, Kelly sizing
  data_sources.py     stats.wnba.com / ESPN fetchers + diagnostics(), all wrapped to fail soft
  stats_engine.py      Rolling averages/hit rates, minutes profile, usage, teammate on/off splits
  matchup.py           Pace/defense ranks, DvP table builder, H2H hit rate, rest/b2b, game environment
  projection.py         Blend + adjustment + distribution model -> predicted probability
  clv.py               Bet logging, outcome recording, calibration report, the learning loop
  ui.py                 ipywidgets dropdown UI (includes a Run Diagnostics button)
WNBA_Prop_Predictor.ipynb   Self-contained Colab notebook (diagnostics cell runs before the UI)
tests/                        Synthetic-data + mocked-response sanity tests (no live network required)
data/bet_log.csv              Seed CLV/outcome log (headers only)
```

## Running tests locally

```
pip install -r requirements.txt
pytest tests/ -v
```
