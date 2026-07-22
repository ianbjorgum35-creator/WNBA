# WNBA Prop Predictor

Compares a model-predicted probability of a player prop hitting against the
sportsbook's implied probability from odds you enter, then learns from
closing-line value (CLV) and actual outcomes to recalibrate itself over time.

## Quick start (running locally)

This is now a web app -- a small FastAPI server plus a static
mobile/desktop-friendly page -- instead of a Colab notebook. Running it
this way only works while your computer is on and reachable, and only from
other devices on the same WiFi network. If you want it reachable from
anywhere (work WiFi, mobile data, etc.), skip to **Deploy to Render**
below instead -- it's the same app either way.

1. Install dependencies (Python 3.10+):
   ```
   pip install -r requirements.txt
   ```
2. Start the server:
   ```
   uvicorn wnba_predictor.webapp:app --host 0.0.0.0 --port 8000
   ```
3. Open it:
   - **On the computer running it:** `http://localhost:8000`
   - **On your phone** (same WiFi network): find the computer's LAN IP
     (e.g. `ipconfig getifaddr en0` on Mac, `ipconfig` on Windows, `hostname
     -I` on Linux) and open `http://<that-ip>:8000` in your phone's
     browser. `--host 0.0.0.0` is what makes it reachable from other
     devices on the network, not just `localhost`.
4. Use the form: pick player team / opponent team / prop type, type in the
   player's name, the line, the odds, the game total and spread. Those
   (plus prop type/line/odds) are the only fields meant to be manual --
   everything else (stats, ranks, home/away, rest days) is supposed to
   auto-fetch.
5. Tap **Auto-Fetch Data**, then **Run Prediction**. If some fields don't
   populate, tap **Run Diagnostics** to see exactly which data source failed
   and why, then fill those specific fields in by hand.
6. Tap **Log This Prediction** to save it for the learning loop. After the
   game, come back, open **Record Outcome for a Past Bet**, refresh the
   pending list, tap the bet, enter the actual result (and closing odds if
   you have them), and tap **Record Outcome**.
7. Once you've logged ~20+ resolved bets, open **Model Learning** and tap
   **Recalibrate Model**.

Locally the app runs with no login by default. If you set `APP_PASSWORD`
while testing locally (to try the login flow before deploying), the
session cookie is HTTPS-only by default -- add `WNBA_INSECURE_COOKIE=1` as
well so it still works over plain `http://localhost`.

The Colab notebook (`WNBA_Prop_Predictor.ipynb`) still works if you prefer
it, but the web app is now the primary way to use this -- it's the same
`wnba_predictor` package underneath, just served over HTTP with a
responsive UI instead of ipywidgets.

## Deploy to Render (access from anywhere, not just home WiFi)

This gets you a `https://something.onrender.com` URL reachable from work
WiFi, mobile data, anywhere -- at the cost of the free tier spinning the
service down after 15 minutes idle (the next request after that takes
~30-50s to wake it back up).

1. Push this repo to GitHub (already done if you're reading this from the
   repo).
2. In the [Render dashboard](https://dashboard.render.com/), **New >
   Blueprint**, connect this GitHub repo. Render reads `render.yaml` at
   the repo root and sets up the web service automatically.
3. Render will prompt for the env vars marked `sync: false` in
   `render.yaml`:
   - **`APP_PASSWORD`** (required) -- the app is reachable by anyone with
     the URL once deployed, so this puts a login screen in front of it.
     Pick anything; you'll type it once per device/browser (the session is
     remembered for 30 days).
   - `WNBA_DRIVE_CREDENTIALS` / `WNBA_DRIVE_TOKEN` -- only if you're using
     Drive sync (see below); leave blank otherwise.
4. Deploy. Once it's up, open the Render URL, enter the password, and use
   it exactly like the local version.

**Persistence on Render's free tier matters more than it does locally**:
the free tier has no persistent disk, so every time the service spins back
up from idle it starts from a fresh copy of whatever's in the git repo --
`data/bet_log.csv` reverts to what's committed, losing anything logged
since. **Set up Google Drive sync (below) if you deploy this**, so logged
predictions and recorded outcomes survive restarts; without it, treat the
deployed copy as a client for occasional access rather than where you
build up bet history.

To make Drive sync work on Render specifically (it can't pop open a
browser for the OAuth consent screen the way it does locally):
1. Do the one-time OAuth consent **locally first**, using
   `scripts/get_drive_token.py` -- it's a small standalone script, so you
   don't need to install this whole project or run the full app just to
   generate a token:
   ```
   pip install google-auth-oauthlib google-api-python-client google-auth-httplib2
   python scripts/get_drive_token.py
   ```
   Run it from a folder with your downloaded `credentials.json` in it (the
   repo root works, or anywhere). It opens a browser for a one-time Google
   consent screen (click **Advanced > Go to \[app name\] (unsafe)** when
   you hit the unverified-app warning -- expected for your own personal
   OAuth client) and writes `data/token.json`.
2. In the Render dashboard, under your service's **Environment > Secret
   Files**, add two files: `credentials.json` and `token.json`, pasting in
   the contents of your local copies. Render mounts these read-only at
   `/etc/secrets/credentials.json` and `/etc/secrets/token.json`.
3. Set the env vars `WNBA_DRIVE_CREDENTIALS=/etc/secrets/credentials.json`
   and `WNBA_DRIVE_TOKEN=/etc/secrets/token.json` (from step 3 above).
4. Redeploy. The app reads its refresh token from that mounted file; since
   the mount is read-only, refreshed access tokens get cached to local
   ephemeral disk instead (handled automatically) rather than failing.

## What it computes

For the player/prop you specify, it gathers (auto-fetch, with every field
manually editable/overridable):

**Player form:** Season / L20 / L10 / L5 averages and hit rates vs. your
line, minutes/game, projected minutes, usage %, starter status, rotation
role, and a teammate usage/minutes "bump" estimate (from on/off splits when
a teammate is out).

**Matchup / game environment:** opponent pace rank, opponent defensive
rank, a DvP-replacement "opponent stat-allowed" rank (see below), game
total, spread, home/away, rest days, back-to-back flag, head-to-head hit
rate vs. this opponent, plus a manual matchup adjustment slider and a
free-text "context lean" you can enter yourself (e.g. an injury note, a
pace mismatch you noticed).

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

## Persisting your history to Google Drive

The web app stores `data/bet_log.csv` and `data/learned_weights.json`
locally by default. To keep them synced to the same `wnba_prop_predictor`
Google Drive folder the Colab notebook used to write to -- so history
survives across devices/reinstalls, not just across sessions on one
machine -- set up Drive OAuth once:

1. In [Google Cloud Console](https://console.cloud.google.com/), create (or
   pick) a project, enable the **Google Drive API**, then create an
   **OAuth client ID** of type **Desktop app**.
2. Download it and save it as `credentials.json` in the repo root (this
   file is gitignored -- never commit it).
3. Start the app and open the **Google Drive Sync** section, or just tap
   **Auto-Fetch Data**/**Log This Prediction** once -- the first Drive call
   opens a browser tab for a one-time Google consent screen. Since this is
   your own unverified OAuth client, click **Advanced > Go to \[app
   name\] (unsafe)** to proceed -- that warning is expected for a personal
   script, not a sign anything is wrong.
4. After that, a refresh token is cached to `data/token.json` (also
   gitignored) so you won't be prompted again on that machine.

Once configured, every logged prediction, recorded outcome, and
recalibration automatically pushes `bet_log.csv` / `learned_weights.json`
back up to the `wnba_prop_predictor` Drive folder. On startup, the app
pulls down whatever's currently in that folder, so bets logged from your
old Colab sessions (or from another device running this app) carry
straight into the model's training data -- nothing needs to be migrated by
hand.

If you skip this setup, the app still works fine with local-only files;
you'll just need to copy `data/bet_log.csv` between machines yourself.

## Data sources and their limits

**Confirmed (via a live diagnostics run from Colab): `stats.wnba.com` is
blocked outright** -- every call to it times out after several seconds,
consistent with it blackholing requests from cloud/datacenter IPs. This
isn't a bug to chase further; the app is built to route around it
entirely via ESPN:

- ESPN's hidden site API is confirmed reachable and is the primary source
  in practice: team/player lookups, rosters, schedules (home/away, rest
  days, back-to-back -- derived automatically, not asked for manually), and
  **player game logs -- confirmed working end-to-end against the real
  payload shape**. Getting there took a few rounds: the label list lives at
  the payload's top level rather than nested inside each category, that
  top level actually carries *two* parallel label lists (a verbose one like
  `"points"` and a short one like `"PTS"`), and several stats (FG, 3PT, FT)
  come back as combined "made-attempted" strings like `"1-6"` that need the
  made count pulled out before they'll parse as numbers. The parser now
  handles all three, and fails loudly (rather than silently returning
  hollow rows) if a future ESPN
  change breaks it again.
- Team pace/defense ranks try ESPN's standings endpoint first (one
  request); confirmed live to not carry scoring stats for this league
  (it's W-L-PCT-only), so this falls through to computing points-for/
  against directly from each team's own schedule results instead -- an
  earlier attempt queried each team's `/statistics` endpoint, but a live
  field-name dump confirmed that endpoint's "defensive" category is the
  team's own defensive box-score stats (steals/blocks/rebounds), not
  points allowed, so there was nothing usable there.
- `stats.wnba.com` (mirrors the stats.nba.com API shape with `LeagueID=10`)
  is tried first for player game logs and team pace/defense ranks, on the
  chance it's reachable from your environment, but is not required --
  everything falls through to the ESPN path above when it isn't.

Since neither host was network-testable from the build sandbox itself,
**run the diagnostics cell/button** any time something isn't populating --
it hits every endpoint independently (including the actual gamelog parser
and the team-ranks fallback) and reports exactly which one is failing and
why (timeout vs. HTTP error vs. an unexpected response shape). Watch the
`espn_*` lines specifically, since those are the paths that actually
matter now. That's also why every auto-fetched field stays a plain
editable box regardless: if a fetch ever fails, type the number in
yourself and keep going rather than being blocked.

**DvP (defense-vs-position) is auto-fetched too, but as a team-level
proxy, not truly position-specific.** True DvP needs a league-wide database
of what every team allows broken out by position, which doesn't exist
through any free endpoint found so far -- building it ourselves would mean
fetching every player's game log on every opposing roster, which is a lot
of fragile requests for a return. Instead, `get_espn_stat_allowed_ranks()`
computes "how much of this prop's stat does this opponent give up per game,
ranked league-wide" from each team's last 8 completed boxscores (reusing
the same schedule data already proven working, extended with a per-game
boxscore lookup). Rank 1 = allows the most = most favorable for the Over,
same convention as before. The tradeoff: it can tell you "this team gives
up a lot of rebounds," not "this team is soft on guards specifically." This
is noticeably slower than the other auto-fetches (~15 teams x up to 8
boxscore requests each to build the full league table), so expect Auto-Fetch
to pause for a bit longer on this step; each boxscore is cached for the
rest of the session since a completed game's stats never change.

The ESPN-derived pace rank is a **proxy** (combined scoring per game), not
true possession-based pace like `stats.wnba.com` would give you --
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
  matchup.py           Pace/defense ranks, opponent stat-allowed (DvP proxy) rank, H2H hit rate, rest/b2b, game environment
  projection.py         Blend + adjustment + distribution model -> predicted probability
  clv.py               Bet logging, outcome recording, calibration report, the learning loop
  drive_sync.py          Optional Google Drive OAuth sync for bet_log.csv / learned_weights.json
  webapp.py               FastAPI backend for the web app (stateless REST API over the modules above)
  ui.py                    ipywidgets dropdown UI, used only by the legacy Colab notebook
static/                        Mobile/desktop-friendly frontend (index.html, app.js, styles.css) served by webapp.py
render.yaml                     Render Blueprint for deploying the web app publicly (see "Deploy to Render")
scripts/get_drive_token.py       Standalone one-time script that generates data/token.json for Drive sync
WNBA_Prop_Predictor.ipynb   Legacy Colab notebook (still works, no longer the primary way to use this)
tests/                        Synthetic-data + mocked-response sanity tests (no live network required)
data/bet_log.csv              CLV/outcome log (carried over from the wnba_prop_predictor Drive folder)
```

## Running tests locally

```
pip install -r requirements.txt
pytest tests/ -v
```
