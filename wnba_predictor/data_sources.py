"""Live data fetchers for player stats, team ranks, injuries, and lineups.

These hit unofficial, reverse-engineered endpoints (stats.wnba.com mirrors
the well-known stats.nba.com API shape with LeagueID="10"; ESPN's hidden
site-api backs espn.com/wnba). They can change or rate-limit without notice,
so every function here:

  1. Wraps its request in a try/except and returns a FetchResult instead of
     raising, so a broken endpoint never crashes the notebook.
  2. Has a corresponding manual-entry field in the UI so the user can always
     override or fill in a value the auto-fetch couldn't get.

This module could not be network-tested in the build sandbox (outbound
requests to stats.wnba.com / espn.com were blocked by the sandbox's egress
policy). Test the fetch cells first when you open the notebook in Colab,
where normal internet access is available.
"""

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import requests

from . import config

WNBA_STATS_BASE = "https://stats.wnba.com/stats"
ESPN_SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
ESPN_CORE_BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba"

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# This is the header set the widely-used `nba_api` project relies on to get
# through stats.nba.com's bot filtering (stats.wnba.com is served by the
# same platform); missing any of these tends to trigger the silent
# hang/blackhole behavior rather than a clean error.
DEFAULT_HEADERS = {
    "User-Agent": _UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Origin": "https://www.wnba.com",
    "Referer": "https://www.wnba.com/",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# ESPN's site API doesn't need (and shouldn't get) the nba-stats-specific
# headers above -- keep a separate, plainer header set for it.
ESPN_HEADERS = {
    "User-Agent": _UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/",
}

ESPN_RETRIES = 2

REQUEST_TIMEOUT = 12
CACHE_TTL_SECONDS = 60 * 30  # 30 minutes

# stats.wnba.com (and the stats.nba.com infrastructure it mirrors) is known to
# hang or silently block requests from cloud/datacenter IPs -- which is what
# Colab runs on -- rather than returning a clean error, and this has now been
# confirmed live (consistent ~12s read-timeouts from a Colab session). A
# cookie warm-up and one retry are kept in case it's ever just a transient
# blip elsewhere, but the per-attempt timeout is shorter than the general
# REQUEST_TIMEOUT so a hard block fails in ~12s total instead of ~28s before
# falling through to the ESPN fallback.
STATS_RETRIES = 2
STATS_TIMEOUT = 6
STATS_RETRY_BACKOFF_SECONDS = 2.0
_wnba_session: Optional["requests.Session"] = None


def _get_wnba_session() -> "requests.Session":
    global _wnba_session
    if _wnba_session is not None:
        return _wnba_session
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    try:
        session.get("https://www.wnba.com/", timeout=8)
    except requests.exceptions.RequestException:
        pass  # warm-up is best-effort; proceed with the session regardless
    _wnba_session = session
    return session


@dataclass
class FetchResult:
    success: bool
    data: Any = None
    source: str = "auto"
    message: str = ""

    @classmethod
    def ok(cls, data, message: str = ""):
        return cls(success=True, data=data, source="auto", message=message)

    @classmethod
    def fail(cls, message: str):
        return cls(success=False, data=None, source="auto", message=message)

    @classmethod
    def manual(cls, data, message: str = "manual entry"):
        return cls(success=True, data=data, source="manual", message=message)


def _cache_path(key: str) -> str:
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in key)
    return os.path.join(config.CACHE_DIR, f"{safe}.json")


def _cache_get(key: str):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > CACHE_TTL_SECONDS:
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _cache_set(key: str, value: Any) -> None:
    try:
        with open(_cache_path(key), "w") as f:
            json.dump(value, f)
    except Exception:
        pass


def _get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
               cache_key: Optional[str] = None, session: Optional["requests.Session"] = None,
               retries: int = 1, timeout: int = REQUEST_TIMEOUT) -> FetchResult:
    if cache_key is not None:
        cached = _cache_get(cache_key)
        if cached is not None:
            return FetchResult.ok(cached, message="from cache")

    getter = session.get if session is not None else requests.get
    last_error = "request failed"
    for attempt in range(retries):
        try:
            resp = getter(url, params=params, headers=headers or DEFAULT_HEADERS, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except requests.exceptions.Timeout as exc:
            last_error = f"timed out (likely blocked/throttled for this IP): {exc}"
            if attempt < retries - 1:
                time.sleep(STATS_RETRY_BACKOFF_SECONDS)
            continue
        except requests.exceptions.RequestException as exc:
            last_error = f"request failed: {exc}"
            if attempt < retries - 1:
                time.sleep(STATS_RETRY_BACKOFF_SECONDS)
            continue
        except ValueError as exc:
            return FetchResult.fail(f"bad JSON response: {exc}")
        else:
            if cache_key is not None:
                _cache_set(cache_key, payload)
            return FetchResult.ok(payload)
    return FetchResult.fail(last_error)


# --------------------------------------------------------------------------
# stats.wnba.com -- player game logs, team dashboards
# --------------------------------------------------------------------------

def _stats_endpoint(endpoint: str, params: dict, cache_key: str) -> FetchResult:
    return _get_json(
        f"{WNBA_STATS_BASE}/{endpoint}", params=params, cache_key=cache_key,
        session=_get_wnba_session(), retries=STATS_RETRIES, timeout=STATS_TIMEOUT,
    )


def _rows_from_resultsets(payload: dict, result_name: Optional[str] = None):
    """stats.wnba.com responses use a resultSets/resultSet envelope with
    parallel headers/rowSet arrays. Flatten the requested one into dicts."""
    result_sets = payload.get("resultSets") or payload.get("resultSet")
    if result_sets is None:
        return []
    if isinstance(result_sets, dict):
        result_sets = [result_sets]
    for rs in result_sets:
        if result_name is None or rs.get("name") == result_name:
            headers = rs["headers"]
            return [dict(zip(headers, row)) for row in rs["rowSet"]]
    return []


def get_team_list(season: str = "2026") -> FetchResult:
    res = _stats_endpoint(
        "leaguedashteamstats",
        {
            "College": "", "Conference": "", "Country": "", "DateFrom": "", "DateTo": "",
            "Division": "", "DraftPick": "", "DraftYear": "", "GameScope": "", "Height": "",
            "LastNGames": 0, "LeagueID": "10", "Location": "", "MeasureType": "Base",
            "Month": 0, "OpponentTeamID": 0, "Outcome": "", "PORound": 0,
            "PaceAdjust": "N", "PerMode": "PerGame", "Period": 0, "PlayerExperience": "",
            "PlayerPosition": "", "PlusMinus": "N", "Rank": "N", "Season": season,
            "SeasonSegment": "", "SeasonType": "Regular Season", "ShotClockRange": "",
            "StarterBench": "", "TeamID": 0, "VsConference": "", "VsDivision": "", "Weight": "",
        },
        cache_key=f"team_list_{season}",
    )
    if not res.success:
        return res
    rows = _rows_from_resultsets(res.data)
    return FetchResult.ok(rows)


def get_team_ranks(season: str = "2026") -> FetchResult:
    """Pace + defensive/offensive rating ranks for every team this season."""
    base_res = _stats_endpoint(
        "leaguedashteamstats",
        {
            "College": "", "Conference": "", "Country": "", "DateFrom": "", "DateTo": "",
            "Division": "", "DraftPick": "", "DraftYear": "", "GameScope": "", "Height": "",
            "LastNGames": 0, "LeagueID": "10", "Location": "", "MeasureType": "Advanced",
            "Month": 0, "OpponentTeamID": 0, "Outcome": "", "PORound": 0,
            "PaceAdjust": "N", "PerMode": "PerGame", "Period": 0, "PlayerExperience": "",
            "PlayerPosition": "", "PlusMinus": "N", "Rank": "Y", "Season": season,
            "SeasonSegment": "", "SeasonType": "Regular Season", "ShotClockRange": "",
            "StarterBench": "", "TeamID": 0, "VsConference": "", "VsDivision": "", "Weight": "",
        },
        cache_key=f"team_ranks_{season}",
    )
    if not base_res.success:
        return base_res
    rows = _rows_from_resultsets(base_res.data)
    ranks = {}
    for row in rows:
        ranks[row.get("TEAM_NAME")] = {
            "pace_rank": row.get("PACE_RANK"),
            "off_rating_rank": row.get("OFF_RATING_RANK"),
            "def_rating_rank": row.get("DEF_RATING_RANK"),
            "net_rating_rank": row.get("NET_RATING_RANK"),
        }
    return FetchResult.ok(ranks)


def get_player_id(player_name: str, season: str = "2026") -> FetchResult:
    res = _stats_endpoint(
        "commonallplayers",
        {"LeagueID": "10", "Season": season, "IsOnlyCurrentSeason": 1},
        cache_key=f"all_players_{season}",
    )
    if not res.success:
        return res
    rows = _rows_from_resultsets(res.data)
    name_lower = player_name.strip().lower()
    for row in rows:
        display_name = (row.get("DISPLAY_FIRST_LAST") or "").strip().lower()
        if display_name == name_lower:
            return FetchResult.ok(row.get("PERSON_ID"))
    # fallback: partial match
    for row in rows:
        display_name = (row.get("DISPLAY_FIRST_LAST") or "").strip().lower()
        if name_lower in display_name or display_name in name_lower:
            return FetchResult.ok(row.get("PERSON_ID"), message=f"partial match: {row.get('DISPLAY_FIRST_LAST')}")
    return FetchResult.fail(f"no player found matching '{player_name}'")


def get_player_gamelog(player_id, season: str = "2026") -> FetchResult:
    """Per-game log for the player: date, opponent, minutes, box score stats."""
    res = _stats_endpoint(
        "playergamelog",
        {"LeagueID": "10", "PlayerID": player_id, "Season": season, "SeasonType": "Regular Season"},
        cache_key=f"gamelog_{player_id}_{season}",
    )
    if not res.success:
        return res
    rows = _rows_from_resultsets(res.data)
    return FetchResult.ok(rows)


def get_player_career_gamelogs(player_id, seasons: list) -> FetchResult:
    """Convenience: pull multiple seasons for larger sample / H2H lookups."""
    all_rows = []
    errors = []
    for season in seasons:
        res = get_player_gamelog(player_id, season)
        if res.success:
            for row in res.data:
                row["SEASON"] = season
            all_rows.extend(res.data)
        else:
            errors.append(f"{season}: {res.message}")
    if not all_rows and errors:
        return FetchResult.fail("; ".join(errors))
    msg = "; ".join(errors) if errors else ""
    return FetchResult.ok(all_rows, message=msg)


def get_player_info(player_id) -> FetchResult:
    res = _stats_endpoint(
        "commonplayerinfo", {"LeagueID": "10", "PlayerID": player_id},
        cache_key=f"player_info_{player_id}",
    )
    if not res.success:
        return res
    rows = _rows_from_resultsets(res.data, result_name="CommonPlayerInfo")
    if not rows:
        return FetchResult.fail("no player info returned")
    return FetchResult.ok(rows[0])


# --------------------------------------------------------------------------
# ESPN -- injuries, rosters/lineups, schedule (for rest days / back-to-backs)
# --------------------------------------------------------------------------

def _espn_endpoint(url: str, params: Optional[dict], cache_key: str) -> FetchResult:
    return _get_json(url, params=params, headers=ESPN_HEADERS, cache_key=cache_key, retries=ESPN_RETRIES)


def get_espn_scoreboard(dates: Optional[str] = None) -> FetchResult:
    """dates format YYYYMMDD. Omit for today's slate."""
    params = {"dates": dates} if dates else None
    return _espn_endpoint(f"{ESPN_SITE_BASE}/scoreboard", params, cache_key=f"espn_scoreboard_{dates}")


def get_espn_team_schedule(espn_team_abbr: str, season: Optional[str] = None) -> FetchResult:
    params = {"season": season} if season else None
    return _espn_endpoint(
        f"{ESPN_SITE_BASE}/teams/{espn_team_abbr}/schedule", params,
        cache_key=f"espn_schedule_{espn_team_abbr}_{season}",
    )


def get_espn_injuries() -> FetchResult:
    """League-wide injury report. ESPN's injuries payload is nested per team."""
    return _espn_endpoint(f"{ESPN_SITE_BASE}/injuries", None, cache_key="espn_injuries")


def get_espn_standings() -> FetchResult:
    return _espn_endpoint(f"{ESPN_SITE_BASE}/standings", None, cache_key="espn_standings")


def _flatten_named_stats(node, out=None) -> dict:
    """Recursively collects every {"name": ..., "value": <number>} leaf from
    a nested ESPN stats payload into a flat {name: value} dict. Used for the
    standings endpoint, which -- confirmed live -- doesn't carry per-category
    splits the way the per-team /statistics endpoint does.
    """
    if out is None:
        out = {}
    if isinstance(node, dict):
        name, value = node.get("name"), node.get("value")
        if name and isinstance(value, (int, float)) and not isinstance(value, bool):
            out.setdefault(name, value)
        for v in node.values():
            _flatten_named_stats(v, out)
    elif isinstance(node, list):
        for item in node:
            _flatten_named_stats(item, out)
    return out


_OFFENSE_KEY_CANDIDATES = ["pointsFor", "avgPointsFor", "avgPoints", "pointsScored", "points", "PF"]
_DEFENSE_KEY_CANDIDATES = [
    "pointsAgainst", "avgPointsAgainst", "avgPointsAllowed", "pointsAllowed", "opponentPoints", "PA",
]
_GAMES_KEY_CANDIDATES = ["gamesPlayed", "GP", "games"]


def _extract_points_for_against(stat_map: dict) -> tuple:
    points_for = next((stat_map[k] for k in _OFFENSE_KEY_CANDIDATES if k in stat_map), None)
    points_against = next((stat_map[k] for k in _DEFENSE_KEY_CANDIDATES if k in stat_map), None)
    games = next((stat_map[k] for k in _GAMES_KEY_CANDIDATES if k in stat_map), None)
    # normalize to per-game if these look like season totals rather than averages
    if points_for is not None and games and points_for > 200:
        points_for = points_for / games
        points_against = points_against / games if points_against is not None else None
    return points_for, points_against


def _team_rows_from_standings() -> list:
    res = get_espn_standings()
    if not res.success:
        return []
    try:
        entries = []
        for child in res.data.get("children", []):
            entries.extend((child.get("standings") or {}).get("entries", []))
        if not entries:
            entries = (res.data.get("standings") or {}).get("entries", [])
        rows = []
        for e in entries:
            team_name = (e.get("team") or {}).get("displayName")
            points_for, points_against = _extract_points_for_against(_flatten_named_stats(e.get("stats", [])))
            if team_name and points_for is not None and points_against is not None:
                rows.append({"team_name": team_name, "points_for_pg": points_for, "points_against_pg": points_against})
        return rows
    except (AttributeError, TypeError):
        return []


def _team_rows_from_schedule_scores() -> list:
    """Second-tier fallback: computes points-for/against directly from each
    team's own completed-game scores via get_espn_team_schedule_parsed --
    the same schedule endpoint already confirmed working for home/away and
    rest-day derivation. This replaced an earlier attempt that queried each
    team's /statistics endpoint: a live field-name dump confirmed that
    endpoint's "defensive" category is the team's own defensive box-score
    stats (steals/blocks/rebounds), not points allowed -- it simply doesn't
    carry opponent-scoring data, so there was nothing to parse there.
    """
    teams_res = get_espn_teams()
    if not teams_res.success:
        return []
    rows = []
    for t in teams_res.data:
        abbr, team_name = t.get("abbreviation"), t.get("displayName")
        if not abbr or not team_name:
            continue
        sched_res = get_espn_team_schedule_parsed(abbr)
        if not sched_res.success:
            continue
        scored_games = [
            g for g in sched_res.data
            if g.get("completed") and g.get("own_score") is not None and g.get("opponent_score") is not None
        ]
        if not scored_games:
            continue
        points_for = sum(g["own_score"] for g in scored_games) / len(scored_games)
        points_against = sum(g["opponent_score"] for g in scored_games) / len(scored_games)
        rows.append({"team_name": team_name, "points_for_pg": points_for, "points_against_pg": points_against})
    return rows


def get_espn_team_ranks_fallback() -> FetchResult:
    """Approximates pace/defense ranks from ESPN when stats.wnba.com is
    unreachable. Defense rank is a genuine points-allowed rank; "pace" here
    is only a rough proxy (combined points per game, offense + defense),
    since ESPN doesn't expose true possession-based pace through these
    endpoints -- directionally useful, not the same number stats.wnba.com
    would give you, so it's flagged as a proxy in the returned dict.

    Tries the standings endpoint first (one request, fast); confirmed live
    to not carry scoring stats for this league, so this falls back to
    computing points-for/against from each team's own schedule results
    (one request per team, but built on data confirmed to actually exist).
    """
    team_rows = _team_rows_from_standings()
    source = "espn_standings_proxy"
    if not team_rows:
        team_rows = _team_rows_from_schedule_scores()
        source = "espn_schedule_scores_proxy"
    if not team_rows:
        return FetchResult.fail(
            "no usable points-for/against found in ESPN standings or schedule scores"
        )

    by_def = sorted(team_rows, key=lambda t: t["points_against_pg"])  # fewest allowed = rank 1
    def_rank = {t["team_name"]: i + 1 for i, t in enumerate(by_def)}
    by_pace = sorted(team_rows, key=lambda t: -(t["points_for_pg"] + t["points_against_pg"]))
    pace_rank = {t["team_name"]: i + 1 for i, t in enumerate(by_pace)}

    ranks = {
        t["team_name"]: {
            "pace_rank": pace_rank[t["team_name"]],
            "def_rating_rank": def_rank[t["team_name"]],
            "off_rating_rank": None,
            "net_rating_rank": None,
            "source": source,
        }
        for t in team_rows
    }
    return FetchResult.ok(ranks)


def _parse_espn_score(raw) -> Optional[float]:
    if isinstance(raw, dict):
        raw = raw.get("value", raw.get("displayValue"))
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def get_espn_team_schedule_parsed(espn_team_abbr: str, season: Optional[str] = None) -> FetchResult:
    """Team schedule flattened into {date, opponent_abbr, is_home, completed,
    own_score, opponent_score} rows. Used to auto-derive home/away and rest
    days instead of asking for them manually, and (via own_score/
    opponent_score on completed games) to compute points-for/against
    directly when neither stats.wnba.com nor ESPN's per-team /statistics
    endpoint has that data available."""
    res = get_espn_team_schedule(espn_team_abbr, season)
    if not res.success:
        return res
    try:
        games = []
        for ev in res.data.get("events", []):
            comps = ev.get("competitions") or []
            if not comps:
                continue
            comp = comps[0]
            date_str = comp.get("date") or ev.get("date")
            if not date_str:
                continue
            game_date = datetime.fromisoformat(date_str.replace("Z", "+00:00")).date()
            is_home, opponent_abbr = None, None
            own_score, opponent_score = None, None
            for c in comp.get("competitors", []):
                team_abbr = (c.get("team") or {}).get("abbreviation")
                score = _parse_espn_score(c.get("score"))
                if team_abbr == espn_team_abbr:
                    is_home = c.get("homeAway") == "home"
                    own_score = score
                else:
                    opponent_abbr = team_abbr
                    opponent_score = score
            completed = bool(((comp.get("status") or {}).get("type") or {}).get("completed"))
            games.append({
                "date": game_date, "opponent_abbr": opponent_abbr,
                "is_home": is_home, "completed": completed,
                "own_score": own_score, "opponent_score": opponent_score,
            })
        if not games:
            return FetchResult.fail("no games parsed from ESPN schedule")
        games.sort(key=lambda g: g["date"])
        return FetchResult.ok(games)
    except (AttributeError, TypeError, KeyError, ValueError) as exc:
        return FetchResult.fail(f"unexpected ESPN schedule shape: {exc}")


def find_scheduled_game(games: list, opponent_abbr: str) -> Optional[dict]:
    """Among parsed schedule rows, pick the closest game against the given
    opponent -- the next upcoming one if there is one, else the most recent
    past meeting."""
    matches = [g for g in games if g["opponent_abbr"] == opponent_abbr]
    if not matches:
        return None
    today = date.today()
    upcoming = [g for g in matches if g["date"] >= today]
    if upcoming:
        return min(upcoming, key=lambda g: g["date"])
    return max(matches, key=lambda g: g["date"])


def game_context(espn_team_abbr: str, opponent_espn_abbr: str, season: Optional[str] = None) -> FetchResult:
    """Auto-derives home/away, rest days, and back-to-back status for the
    next (or most recent) game between these two teams, so the user doesn't
    have to enter them by hand."""
    schedule_res = get_espn_team_schedule_parsed(espn_team_abbr, season)
    if not schedule_res.success:
        return schedule_res
    games = schedule_res.data
    target = find_scheduled_game(games, opponent_espn_abbr)
    if target is None:
        return FetchResult.fail(f"no scheduled game found against {opponent_espn_abbr}")
    prior_dates = [g["date"] for g in games if g["date"] < target["date"]]
    rest_days = (target["date"] - max(prior_dates)).days - 1 if prior_dates else None
    return FetchResult.ok({
        "game_date": target["date"],
        "is_home": target["is_home"],
        "rest_days": rest_days,
        "is_b2b": rest_days is not None and rest_days <= 0,
    })


def get_espn_teams() -> FetchResult:
    """Full team index (id, ESPN's own abbreviation, names). Used to resolve
    our team names to *ESPN's* abbreviation rather than assuming it matches
    ours -- e.g. Connecticut could be CONN on ESPN vs. our CON -- since a
    wrong guess there would silently 404 every other ESPN call for that
    team."""
    res = _espn_endpoint(f"{ESPN_SITE_BASE}/teams", None, cache_key="espn_teams")
    if not res.success:
        return res
    try:
        teams = res.data["sports"][0]["leagues"][0]["teams"]
        return FetchResult.ok([t["team"] for t in teams])
    except (KeyError, IndexError, TypeError) as exc:
        return FetchResult.fail(f"unexpected ESPN teams shape: {exc}")


def resolve_espn_team_abbr(team_full_name: str) -> FetchResult:
    teams_res = get_espn_teams()
    if not teams_res.success:
        return teams_res
    name_lower = team_full_name.strip().lower()
    for t in teams_res.data:
        display = (t.get("displayName") or "").strip().lower()
        if display == name_lower:
            return FetchResult.ok(t.get("abbreviation"))
    for t in teams_res.data:
        display = (t.get("displayName") or "").strip().lower()
        if name_lower in display or display in name_lower:
            return FetchResult.ok(t.get("abbreviation"), message=f"partial match: {display}")
    return FetchResult.fail(f"no ESPN team found matching '{team_full_name}'")


ESPN_COMMON_BASE = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/wnba"

# Maps the stat label strings ESPN's gamelog API uses onto our normalized
# column names. ESPN doesn't document this endpoint, so this list covers the
# label spellings seen in the equivalent NBA gamelog payload (WNBA is served
# by the same platform); unmatched labels are simply dropped rather than
# raising, since this fallback must degrade safely if ESPN changes the shape.
_ESPN_STAT_LABEL_MAP = {
    "MIN": "MIN", "PTS": "PTS", "REB": "REB", "AST": "AST",
    "STL": "STL", "BLK": "BLK", "TO": "TOV", "TOV": "TOV",
    "3PM": "FG3M", "3PT": "FG3M",
}


def get_espn_player_id(player_name: str, espn_team_abbr: str) -> FetchResult:
    """Resolve an athlete ID from a team roster by (fuzzy) name match."""
    roster_res = get_espn_team_roster(espn_team_abbr)
    if not roster_res.success:
        return roster_res
    try:
        athletes = roster_res.data.get("athletes", [])
        # ESPN sometimes groups the roster by position; flatten if so.
        flat = []
        for a in athletes:
            if "items" in a:
                flat.extend(a["items"])
            else:
                flat.append(a)
        name_lower = player_name.strip().lower()
        for a in flat:
            display = (a.get("displayName") or a.get("fullName") or "").strip().lower()
            if display == name_lower:
                return FetchResult.ok(a.get("id"))
        for a in flat:
            display = (a.get("displayName") or a.get("fullName") or "").strip().lower()
            if name_lower in display or display in name_lower:
                return FetchResult.ok(a.get("id"), message=f"partial match: {display}")
    except (AttributeError, TypeError, KeyError) as exc:
        return FetchResult.fail(f"unexpected roster shape: {exc}")
    return FetchResult.fail(f"no player found matching '{player_name}' on {espn_team_abbr} roster")


def _extract_event_stats(event: dict, labels: list) -> dict:
    """A single game's stats can come back either as a parallel array
    (event["stats"] = ["32", "18", ...] aligned to the category's label
    list) or as a list of {"name"/"abbreviation": ..., "value"/"displayValue": ...}
    dicts (the shape confirmed live for ESPN's per-team /statistics
    endpoint). Try both rather than assuming one, since guessing wrong here
    previously produced rows with correct dates/opponents but silently
    empty stat columns -- a failure that looked like success.
    """
    stats = event.get("stats", [])
    row = {}

    if stats and all(isinstance(s, dict) for s in stats):
        for s in stats:
            key = s.get("name") or s.get("abbreviation")
            mapped = _ESPN_STAT_LABEL_MAP.get(str(key).upper()) if key else None
            if not mapped:
                continue
            value = s.get("value", s.get("displayValue"))
            row[mapped] = value
        return row

    col_index = {}
    for i, label in enumerate(labels):
        mapped = _ESPN_STAT_LABEL_MAP.get(str(label).upper())
        if mapped:
            col_index[mapped] = i
    for col, idx in col_index.items():
        if idx < len(stats):
            row[col] = stats[idx]
    return row


def get_espn_player_gamelog(athlete_id) -> FetchResult:
    """Fallback player game log via ESPN's (undocumented) gamelog endpoint,
    used when stats.wnba.com is unreachable. Best-effort: returns
    FetchResult.fail rather than raising if the response doesn't match the
    expected shape, so a schema change here degrades safely to manual entry.
    """
    res = _espn_endpoint(
        f"{ESPN_COMMON_BASE}/athletes/{athlete_id}/gamelog", None,
        cache_key=f"espn_gamelog_{athlete_id}",
    )
    if not res.success:
        return res

    try:
        payload = res.data
        events_meta = payload.get("events", {})
        rows = []
        for season_type in payload.get("seasonTypes", []):
            for category in season_type.get("categories", []):
                labels = category.get("labels") or category.get("names") or []
                for event in category.get("events", []):
                    event_id = event.get("eventId") or event.get("id")
                    meta = events_meta.get(str(event_id), {}) if isinstance(events_meta, dict) else {}
                    opponent = (meta.get("opponent") or {}).get("abbreviation", "")
                    at_vs = meta.get("atVs", "vs")
                    row = {
                        "GAME_DATE": meta.get("gameDate"),
                        "MATCHUP": f"{at_vs} {opponent}".strip(),
                    }
                    row.update(_extract_event_stats(event, labels))
                    rows.append(row)
        if not rows:
            return FetchResult.fail("ESPN gamelog returned no parsable rows")
        stat_cols = set(_ESPN_STAT_LABEL_MAP.values())
        if not any(stat_cols & row.keys() for row in rows):
            return FetchResult.fail(
                "ESPN gamelog returned rows (dates/opponents parsed fine) but no recognizable "
                "stat columns -- the label/stat shape doesn't match what this parser expects"
            )
        return FetchResult.ok(rows)
    except (AttributeError, TypeError, KeyError, IndexError) as exc:
        return FetchResult.fail(f"unexpected ESPN gamelog shape: {exc}")


def dump_espn_gamelog_shape(athlete_id) -> dict:
    """Introspection helper for get_espn_player_gamelog: returns the raw
    payload's top-level structure (season type / category / event keys and
    a sample event) so a parsing mismatch can be fixed from real structure
    instead of another guess.
    """
    res = _espn_endpoint(
        f"{ESPN_COMMON_BASE}/athletes/{athlete_id}/gamelog", None,
        cache_key=f"espn_gamelog_{athlete_id}",
    )
    if not res.success:
        return {"error": res.message}
    payload = res.data
    if not isinstance(payload, dict):
        return {"top_level_type": str(type(payload))}

    summary = {"top_level_keys": sorted(payload.keys())}
    season_types = payload.get("seasonTypes") or []
    if season_types:
        st0 = season_types[0]
        summary["season_type_keys"] = sorted(st0.keys())
        categories = st0.get("categories") or []
        if categories:
            cat0 = categories[0]
            summary["category_keys"] = sorted(cat0.keys())
            summary["category_labels"] = cat0.get("labels") or cat0.get("names")
            events = cat0.get("events") or []
            if events:
                summary["sample_event"] = events[0]

    events_meta = payload.get("events")
    if isinstance(events_meta, dict) and events_meta:
        first_key = next(iter(events_meta))
        summary["sample_events_meta_entry"] = events_meta[first_key]

    return summary


def get_espn_team_roster(espn_team_abbr: str) -> FetchResult:
    return _espn_endpoint(
        f"{ESPN_SITE_BASE}/teams/{espn_team_abbr}/roster", None,
        cache_key=f"espn_roster_{espn_team_abbr}",
    )


def compute_rest_days(team_schedule_rows: list, game_date) -> FetchResult:
    """Given a sorted list of a team's game dates (datetime.date) and the
    upcoming game date, return (rest_days, is_back_to_back)."""
    prior_games = [d for d in team_schedule_rows if d < game_date]
    if not prior_games:
        return FetchResult.fail("no prior games found to compute rest")
    last_game = max(prior_games)
    rest_days = (game_date - last_game).days - 1
    return FetchResult.ok({"rest_days": rest_days, "is_b2b": rest_days <= 0})


# --------------------------------------------------------------------------
# Diagnostics -- run this when auto-fetch isn't pulling anything. It reports
# exactly which host/endpoint is failing and why (timeout vs. HTTP error vs.
# an unexpected response shape), instead of leaving you guessing. None of
# this module could be network-tested from the environment it was built in
# (its egress policy blocked both stats.wnba.com and espn.com), so the
# diagnostic output from a real run is the only way to pin down what's
# actually broken.
# --------------------------------------------------------------------------

def diagnostics(player_name: Optional[str] = None, player_team_name: Optional[str] = None) -> dict:
    """player_team_name is the full team name (e.g. "Indiana Fever"), matching
    what the UI's team dropdown holds -- not an abbreviation."""
    report = {}

    def _check(name, fn):
        """Runs fn(), records a report entry, and returns the underlying
        value on success (FetchResult.data, or the raw return value) so
        callers can chain off it without re-fetching."""
        start = time.time()
        try:
            result = fn()
            elapsed = round(time.time() - start, 2)
            if isinstance(result, FetchResult):
                report[name] = {
                    "ok": result.success,
                    "elapsed_sec": elapsed,
                    "detail": result.message if not result.success else "ok",
                    "sample": str(result.data)[:200] if result.success else None,
                }
                return result.data if result.success else None
            report[name] = {"ok": True, "elapsed_sec": elapsed, "sample": str(result)[:200]}
            return result
        except Exception as exc:  # noqa: BLE001 -- diagnostics must never raise
            report[name] = {
                "ok": False, "elapsed_sec": round(time.time() - start, 2),
                "detail": f"{type(exc).__name__}: {exc}",
            }
            return None

    print("Running data-source diagnostics -- this can take up to ~90s if some hosts are blocked...")

    _check("stats_wnba_reachability (team ranks)", get_team_ranks)
    _check("espn_reachability (scoreboard)", get_espn_scoreboard)
    _check("espn_teams", get_espn_teams)

    espn_abbr = None
    if player_team_name:
        espn_abbr = _check("espn_resolve_team_abbr", lambda: resolve_espn_team_abbr(player_team_name))
        if espn_abbr:
            _check("espn_team_roster", lambda: get_espn_team_roster(espn_abbr))
            _check("espn_team_schedule_parsed", lambda: get_espn_team_schedule_parsed(espn_abbr))

    _check("espn_team_ranks_fallback (standings/schedule-scores)", get_espn_team_ranks_fallback)

    if player_name:
        _check("stats_wnba_player_id", lambda: get_player_id(player_name))
        if espn_abbr:
            espn_player_id = _check("espn_player_id", lambda: get_espn_player_id(player_name, espn_abbr))
            if espn_player_id:
                gamelog_ok = _check("espn_player_gamelog", lambda: get_espn_player_gamelog(espn_player_id))
                if gamelog_ok is None:
                    shape = dump_espn_gamelog_shape(espn_player_id)
                    report["espn_gamelog_shape_dump"] = {
                        "ok": True, "elapsed_sec": 0.0, "detail": str(shape)[:4000],
                    }

    for name, entry in report.items():
        status = "OK" if entry["ok"] else "FAILED"
        detail = entry.get("detail", entry.get("sample", ""))
        print(f"  [{status}] {name} ({entry['elapsed_sec']}s): {detail}")

    return report
