"""FastAPI backend for the mobile/desktop-friendly WNBA Prop Predictor app.

Replaces the Colab + ipywidgets UI (wnba_predictor.ui) with a stateless REST
API consumed by static/index.html. Stateless by design, same philosophy as
the notebook: auto-fetch fills in a flat dict of fields, the client holds
and can edit every one of them, and /predict just takes whatever the client
currently has -- a broken/rate-limited data source never blocks a
prediction, you just type the number in.

Run with:  uvicorn wnba_predictor.webapp:app --host 0.0.0.0 --port 8000
"""

import hashlib
import hmac
import os
import traceback
from typing import Optional

from fastapi import Body, FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import clv, config, data_sources, drive_sync, matchup, projection, stats_engine
from .odds import edge as edge_fn
from .odds import implied_prob_from_american, kelly_stake

TEAM_ABBR = {t["name"]: t["abbr"] for t in config.WNBA_TEAMS}

BET_LOG_PATH = os.environ.get("WNBA_BET_LOG_PATH", config.BET_LOG_PATH)
WEIGHTS_PATH = os.environ.get("WNBA_WEIGHTS_PATH", config.LEARNED_WEIGHTS_PATH)

app = FastAPI(title="WNBA Prop Predictor")


@app.on_event("startup")
def _startup():
    clv.ensure_log_exists(BET_LOG_PATH)
    try:
        drive_sync.sync_down(BET_LOG_PATH, WEIGHTS_PATH)
    except Exception:
        pass  # best-effort; local files are still usable


# ----------------------------------------------------------------------
# Password gate (single shared password -- this is a personal tool, not a
# multi-user app). Unset APP_PASSWORD to run with no auth (e.g. local dev).
# The session cookie is a fixed HMAC of the password itself, so it survives
# server restarts without needing a separately-managed secret, and a wrong
# guess can't forge it without knowing APP_PASSWORD.
# ----------------------------------------------------------------------
APP_PASSWORD = os.environ.get("APP_PASSWORD")
SESSION_COOKIE = "wnba_session"
# Cookie defaults to HTTPS-only (correct on Render, which terminates TLS at
# its proxy). Set WNBA_INSECURE_COOKIE=1 only for local http:// testing.
_COOKIE_SECURE = os.environ.get("WNBA_INSECURE_COOKIE") != "1"

LOGIN_PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WNBA Prop Predictor -- Sign in</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: #f5f6f8; display: flex; align-items: center; justify-content: center;
          height: 100vh; margin: 0; }}
  form {{ background: #fff; border: 1px solid #dcdfe4; border-radius: 10px; padding: 1.5rem;
          width: 260px; }}
  h1 {{ font-size: 1.1rem; margin: 0 0 1rem; }}
  input {{ width: 100%; padding: 0.55rem; border-radius: 8px; border: 1px solid #dcdfe4;
           box-sizing: border-box; font-size: 1rem; }}
  button {{ width: 100%; margin-top: 0.8rem; padding: 0.6rem; border-radius: 8px; border: none;
            background: #2563eb; color: #fff; font-size: 1rem; }}
  p.error {{ color: #c0362c; font-size: 0.85rem; margin: 0.5rem 0 0; }}
</style></head>
<body>
<form method="post" action="/login">
  <h1>WNBA Prop Predictor</h1>
  <input type="password" name="password" placeholder="Password" autofocus required>
  {error_html}
  <button type="submit">Sign in</button>
</form>
</body></html>"""


def _expected_session_token() -> str:
    return hmac.new(APP_PASSWORD.encode(), b"wnba-session-v1", hashlib.sha256).hexdigest()


def _authenticated(request: Request) -> bool:
    if not APP_PASSWORD:
        return True
    cookie = request.cookies.get(SESSION_COOKIE, "")
    return hmac.compare_digest(cookie, _expected_session_token())


@app.middleware("http")
async def password_gate(request: Request, call_next):
    if not APP_PASSWORD or request.url.path in ("/login", "/favicon.ico"):
        return await call_next(request)
    if _authenticated(request):
        return await call_next(request)
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return RedirectResponse(url="/login")


@app.get("/login", response_class=HTMLResponse)
def login_form(error: Optional[str] = None):
    error_html = '<p class="error">Wrong password.</p>' if error else ""
    return LOGIN_PAGE.format(error_html=error_html)


@app.post("/login")
def login_submit(password: str = Form(...)):
    if APP_PASSWORD and hmac.compare_digest(password, APP_PASSWORD):
        resp = RedirectResponse(url="/", status_code=303)
        resp.set_cookie(
            SESSION_COOKIE, _expected_session_token(),
            httponly=True, samesite="lax", secure=_COOKIE_SECURE, max_age=60 * 60 * 24 * 30,
        )
        return resp
    return RedirectResponse(url="/login?error=1", status_code=303)


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login")
    resp.delete_cookie(SESSION_COOKIE)
    return resp


@app.get("/favicon.ico")
def favicon():
    return HTMLResponse(status_code=204, content="")


# ----------------------------------------------------------------------
# Static config
# ----------------------------------------------------------------------
@app.get("/api/config")
def get_config():
    return {
        "teams": config.TEAM_NAMES,
        "prop_types": [{"key": p.key, "label": p.label} for p in config.PROP_TYPES],
        "directions": config.DIRECTIONS,
    }


# ----------------------------------------------------------------------
# Auto-fetch
# ----------------------------------------------------------------------
def _fetch_player_data(player_name, player_team, opponent_team, prop_key, line, direction):
    fields, messages = {}, []
    gamelog_res = None
    source_used = "stats_wnba"

    player_res = data_sources.get_player_id(player_name)
    if player_res.success:
        gamelog_res = data_sources.get_player_gamelog(player_res.data)
        if not gamelog_res.success:
            messages.append(f"stats.wnba.com game log fetch failed: {gamelog_res.message}")
    else:
        messages.append(f"stats.wnba.com player lookup failed: {player_res.message}")

    espn_opponent_abbr = None
    if gamelog_res is None or not gamelog_res.success:
        messages.append("Trying ESPN as a fallback source...")
        espn_team_res = data_sources.resolve_espn_team_abbr(player_team)
        if not espn_team_res.success:
            messages.append(f"Could not resolve ESPN team abbreviation: {espn_team_res.message}")
            return fields, messages
        espn_id_res = data_sources.get_espn_player_id(player_name, espn_team_res.data)
        if not espn_id_res.success:
            messages.append(f"ESPN player lookup also failed: {espn_id_res.message}")
            return fields, messages
        gamelog_res = data_sources.get_espn_player_gamelog(espn_id_res.data)
        if not gamelog_res.success:
            messages.append(f"ESPN game log fetch also failed: {gamelog_res.message}")
            return fields, messages
        source_used = "espn"
        espn_opp_res = data_sources.resolve_espn_team_abbr(opponent_team)
        espn_opponent_abbr = espn_opp_res.data if espn_opp_res.success else None
        messages.append("Loaded game log from ESPN fallback.")

    df = stats_engine.normalize_gamelog(gamelog_res.data)
    rolling = stats_engine.rolling_windows(df, prop_key, line, direction)
    minutes = stats_engine.minutes_profile(df)
    usage = stats_engine.usage_profile(df)

    fields.update({
        "season_avg": rolling.get("season_avg") or 0.0,
        "l20_avg": rolling.get("l20_avg") or 0.0,
        "l10_avg": rolling.get("l10_avg") or 0.0,
        "l5_avg": rolling.get("l5_avg") or 0.0,
        "l20_hit": rolling.get("l20_hit_rate") or 0.0,
        "l10_hit": rolling.get("l10_hit_rate") or 0.0,
        "l5_hit": rolling.get("l5_hit_rate") or 0.0,
        "season_hit": rolling.get("season_hit_rate") or 0.0,
        "season_games": rolling.get("season_games") or 0,
        "min_per_game": minutes.get("season_avg_min") or 0.0,
        "proj_minutes": minutes.get("l5_avg_min") or minutes.get("season_avg_min") or 0.0,
        "usage_pct": usage.get("l10_usage_pct") or usage.get("season_usage_pct") or 0.0,
        "starter": minutes.get("rotation_role") == "Starter",
        "rotation_role": minutes.get("rotation_role") or "",
    })
    messages.append(f"Player stats loaded from {len(df)} games.")

    if source_used == "espn":
        espn_own_res = data_sources.resolve_espn_team_abbr(player_team)
        own_abbr = espn_own_res.data if espn_own_res.success else TEAM_ABBR.get(player_team)
        opp_abbr = espn_opponent_abbr or TEAM_ABBR.get(opponent_team)
    else:
        own_abbr = TEAM_ABBR.get(player_team)
        opp_abbr = TEAM_ABBR.get(opponent_team)

    h2h = matchup.h2h_hit_rate(df, own_abbr, opp_abbr, prop_key, line, direction)
    if h2h.get("h2h_hit_rate") is not None:
        fields["h2h_hit"] = h2h["h2h_hit_rate"]
    messages.append(f"H2H vs {opponent_team}: {h2h.get('h2h_games', 0)} games found.")
    return fields, messages


def _fetch_matchup_data(player_team, opponent_team, prop_key):
    fields, messages = {}, []
    ranks_res = data_sources.get_team_ranks()
    used_fallback = False
    if not ranks_res.success:
        messages.append(f"stats.wnba.com team ranks fetch failed: {ranks_res.message}")
        messages.append("Trying ESPN as a fallback (pace becomes a rough proxy)...")
        ranks_res = data_sources.get_espn_team_ranks_fallback()
        used_fallback = True
        if not ranks_res.success:
            messages.append(f"ESPN fallback also failed: {ranks_res.message}")

    if ranks_res.success:
        entry = matchup.team_pace_and_defense(ranks_res.data, opponent_team)
        if entry.get("pace_rank") is not None:
            fields["pace_rank"] = entry["pace_rank"]
        if entry.get("def_rating_rank") is not None:
            fields["def_rank"] = entry["def_rating_rank"]
        source = entry.get("source")
        note = f" ({source}, proxy not true pace)" if used_fallback and source else ""
        messages.append(f"Opponent ranks loaded{note}: {entry}")

    messages.append("Computing Opp Stat Allowed Rank (team-level proxy, queries all 15 teams -- slower)...")
    stat_ranks_res = data_sources.get_espn_stat_allowed_ranks(prop_key)
    if not stat_ranks_res.success:
        messages.append(f"Opp Stat Allowed Rank fetch failed: {stat_ranks_res.message}")
    else:
        stat_entry = matchup.team_stat_allowed_rank(stat_ranks_res.data, opponent_team)
        if stat_entry.get("stat_allowed_rank") is not None:
            fields["dvp_rank"] = stat_entry["stat_allowed_rank"]
            messages.append(f"Opp Stat Allowed Rank loaded: {stat_entry}")
        else:
            messages.append(f"Opponent not found in stat-allowed table: {stat_entry}")

    own_espn_res = data_sources.resolve_espn_team_abbr(player_team)
    opp_espn_res = data_sources.resolve_espn_team_abbr(opponent_team)
    if not (own_espn_res.success and opp_espn_res.success):
        messages.append("Could not resolve ESPN team abbreviations; leaving home/away, rest, b2b manual.")
        return fields, messages
    context_res = data_sources.game_context(own_espn_res.data, opp_espn_res.data)
    if not context_res.success:
        messages.append(f"Game context fetch failed: {context_res.message}")
        return fields, messages
    ctx = context_res.data
    if ctx.get("is_home") is not None:
        fields["home_away"] = "Home" if ctx["is_home"] else "Away"
    if ctx.get("rest_days") is not None:
        fields["rest_days"] = ctx["rest_days"]
    if ctx.get("is_b2b") is not None:
        fields["is_b2b"] = ctx["is_b2b"]
    messages.append(f"Game context loaded: {ctx.get('game_date')}, "
                     f"{'Home' if ctx.get('is_home') else 'Away'}, rest_days={ctx.get('rest_days')}.")
    return fields, messages


@app.post("/api/autofetch")
def autofetch(body: dict = Body(...)):
    player_name = body["player_name"]
    player_team = body["player_team"]
    opponent_team = body["opponent_team"]
    prop_key = body["prop_key"]
    line = body.get("line", 0) or 0
    direction = body.get("direction", "Over")

    fields, messages = {}, []
    try:
        f, m = _fetch_player_data(player_name, player_team, opponent_team, prop_key, line, direction)
        fields.update(f)
        messages.extend(m)
    except Exception:
        messages.append("Player stat fetch failed:\n" + traceback.format_exc())

    try:
        f, m = _fetch_matchup_data(player_team, opponent_team, prop_key)
        fields.update(f)
        messages.extend(m)
    except Exception:
        messages.append("Matchup data fetch failed:\n" + traceback.format_exc())

    return {"fields": fields, "messages": messages}


@app.get("/api/diagnostics")
def diagnostics(player_name: Optional[str] = None, player_team_name: Optional[str] = None):
    report = data_sources.diagnostics(player_name=player_name or None, player_team_name=player_team_name or None)
    return report


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
def _build_inputs(body: dict) -> projection.ProjectionInputs:
    prop_key = body["prop_key"]
    rolling = {
        "season_avg": body.get("season_avg") or None,
        "l20_avg": body.get("l20_avg") or None,
        "l10_avg": body.get("l10_avg") or None,
        "l5_avg": body.get("l5_avg") or None,
        "l20_hit_rate": body.get("l20_hit"),
        "l10_hit_rate": body.get("l10_hit"),
        "l5_hit_rate": body.get("l5_hit"),
        "season_hit_rate": body.get("season_hit", body.get("l20_hit")),
        "season_games": body.get("season_games") or 10,
        "season_std": None, "l20_std": None, "l10_std": None,
    }
    proj_minutes = body.get("proj_minutes") or body.get("min_per_game") or None
    minutes_profile = {
        "season_avg_min": body.get("min_per_game") or None,
        "l10_avg_min": body.get("min_per_game") or None,
        "l5_avg_min": proj_minutes,
        "rotation_role": body.get("rotation_role") or ("Starter" if body.get("starter") else "Rotation"),
    }
    usage_profile = {"l10_usage_pct": body.get("usage_pct") or None}

    weights_res = clv.update_blend_weights(BET_LOG_PATH)
    blend_weights = weights_res.get("weights", config.DEFAULT_BLEND_WEIGHTS)

    return projection.ProjectionInputs(
        prop_key=prop_key,
        direction=body.get("direction", "Over"),
        line=float(body.get("line") or 0),
        rolling=rolling,
        minutes_profile=minutes_profile,
        projected_minutes=proj_minutes,
        usage_profile=usage_profile,
        opponent_pace_rank=body.get("pace_rank") or None,
        opponent_def_rating_rank=body.get("def_rank") or None,
        dvp_rank=body.get("dvp_rank") or None,
        game_total=body.get("game_total") or None,
        spread=body.get("spread"),
        is_home=(body.get("home_away") == "Home"),
        rest_days=body.get("rest_days"),
        is_b2b=bool(body.get("is_b2b")),
        h2h={"h2h_hit_rate": body.get("h2h_hit") or None},
        teammate_usage_bump_pct=body.get("teammate_usage_bump") or None,
        teammate_minutes_bump_pct=body.get("teammate_minutes_bump") or None,
        manual_matchup_adjustment=body.get("manual_adj", 0.0) or 0.0,
        context_lean=body.get("context_lean", 0.0) or 0.0,
        context_note=body.get("context_note", "") or "",
        lineup_confirmed=bool(body.get("lineup_confirmed", True)),
        blend_weights=blend_weights,
    )


@app.post("/api/predict")
def predict(body: dict = Body(...)):
    inputs = _build_inputs(body)
    calibrator = clv.load_calibrator(WEIGHTS_PATH)
    result = projection.run_prediction(inputs, calibrator=calibrator)

    odds = float(body.get("odds", -110) or -110)
    implied = implied_prob_from_american(odds)
    model_edge = edge_fn(result.predicted_prob, implied)
    kelly = kelly_stake(result.predicted_prob, odds)

    prop_label = next((p.label for p in config.PROP_TYPES if p.key == inputs.prop_key), inputs.prop_key)

    record = {
        "player_name": body.get("player_name", ""),
        "player_team": body.get("player_team", ""),
        "opponent_team": body.get("opponent_team", ""),
        "prop_type": prop_label,
        "direction": inputs.direction,
        "line": inputs.line,
        "odds_open": odds,
        "implied_prob_open": implied,
        "predicted_prob": result.predicted_prob,
        "raw_predicted_prob": result.raw_predicted_prob,
        "edge": model_edge,
        "season_hit_rate": inputs.rolling.get("season_hit_rate"),
        "l20_hit_rate": inputs.rolling.get("l20_hit_rate"),
        "l10_hit_rate": inputs.rolling.get("l10_hit_rate"),
        "l5_hit_rate": inputs.rolling.get("l5_hit_rate"),
        "blend_weights_json": _json(inputs.blend_weights),
        "adjustments_json": _json(result.adjustments),
        "breakdown_json": _json(result.breakdown),
        "confidence": result.confidence,
        "lineup_confirmed": inputs.lineup_confirmed,
        "manual_matchup_adjustment": inputs.manual_matchup_adjustment,
        "context_lean": inputs.context_lean,
    }

    return {
        "raw_predicted_prob": result.raw_predicted_prob,
        "predicted_prob": result.predicted_prob,
        "projected_mean": result.projected_mean,
        "projected_std": result.projected_std,
        "adjustments": result.adjustments,
        "total_adjustment": result.total_adjustment,
        "confidence": result.confidence,
        "confidence_score": result.confidence_score,
        "breakdown": result.breakdown,
        "implied_prob": implied,
        "edge": model_edge,
        "kelly_full": kelly.full_kelly,
        "kelly_quarter": kelly.quarter_kelly,
        "record": record,
    }


def _json(obj):
    import json
    return json.dumps(obj, default=str)


def _json_safe(obj):
    """Round-trips through json.dumps(default=str) so stray numpy/pandas
    scalar types (int64, bool_, Timestamp, pandas Interval, ...) never trip
    up FastAPI's response encoder, then strips NaN/Infinity (valid in
    Python's json.dumps by default, but not in the strict JSON Starlette
    emits) since pandas leaves those in empty-bin aggregates."""
    import json
    import math

    def _strip_nonfinite(x):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        if isinstance(x, dict):
            return {k: _strip_nonfinite(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_strip_nonfinite(v) for v in x]
        return x

    return _strip_nonfinite(json.loads(json.dumps(obj, default=str)))


# ----------------------------------------------------------------------
# Bet logging / outcomes / learning
# ----------------------------------------------------------------------
@app.post("/api/log")
def log_bet(record: dict = Body(...)):
    bet_id = clv.log_bet(record, BET_LOG_PATH)
    drive_status = _sync_up()
    return {"bet_id": bet_id, "drive": drive_status}


@app.get("/api/pending")
def pending_bets():
    df = clv.pending_bets(BET_LOG_PATH)
    if df.empty:
        return []
    cols = ["bet_id", "logged_at", "player_name", "player_team", "opponent_team",
            "prop_type", "direction", "line", "odds_open", "predicted_prob"]
    return _json_safe(df[cols].to_dict(orient="records"))


@app.post("/api/record_outcome")
def record_outcome(body: dict = Body(...)):
    try:
        updated = clv.record_outcome(
            body["bet_id"],
            float(body["actual_stat_value"]),
            closing_odds=body.get("closing_odds") or None,
            path=BET_LOG_PATH,
            notes=body.get("notes", "") or "",
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    drive_status = _sync_up()
    return {"updated": _json_safe(updated), "drive": drive_status}


@app.get("/api/calibration")
def calibration():
    return _json_safe(clv.calibration_report(BET_LOG_PATH))


@app.post("/api/recalibrate")
def recalibrate():
    report = clv.calibration_report(BET_LOG_PATH)
    fit = clv.fit_recalibration(BET_LOG_PATH, weights_path=WEIGHTS_PATH)
    weights = clv.update_blend_weights(BET_LOG_PATH)
    drive_status = _sync_up()
    return _json_safe({"calibration": report, "recalibration": fit, "blend_weights": weights, "drive": drive_status})


def _sync_up():
    try:
        return drive_sync.sync_up(BET_LOG_PATH, WEIGHTS_PATH)
    except Exception as e:
        return {"enabled": True, "connected": False, "message": f"Drive push failed: {e}"}


# ----------------------------------------------------------------------
# Drive sync controls
# ----------------------------------------------------------------------
@app.get("/api/drive/status")
def drive_status():
    return drive_sync.status()


@app.post("/api/drive/pull")
def drive_pull():
    return drive_sync.sync_down(BET_LOG_PATH, WEIGHTS_PATH)


@app.post("/api/drive/push")
def drive_push():
    return _sync_up()


# ----------------------------------------------------------------------
# Static frontend
# ----------------------------------------------------------------------
_STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
