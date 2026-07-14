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
from typing import Any, Optional

import requests

from . import config

WNBA_STATS_BASE = "https://stats.wnba.com/stats"
ESPN_SITE_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba"
ESPN_CORE_BASE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/wnba"

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.wnba.com",
    "Referer": "https://www.wnba.com/",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

REQUEST_TIMEOUT = 12
CACHE_TTL_SECONDS = 60 * 30  # 30 minutes

# stats.wnba.com (and the stats.nba.com infrastructure it mirrors) is known to
# hang or silently block requests from cloud/datacenter IPs -- which is what
# Colab runs on -- rather than returning a clean error. Two mitigations below:
# a cookie warm-up hit against the public wnba.com site (some bot filters key
# off having a same-origin cookie) and a couple of retries so a one-off
# network blip doesn't look identical to a hard block.
STATS_RETRIES = 2
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
        session=_get_wnba_session(), retries=STATS_RETRIES,
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


def get_team_list(season: str = "2025") -> FetchResult:
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


def get_team_ranks(season: str = "2025") -> FetchResult:
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


def get_player_id(player_name: str, season: str = "2025") -> FetchResult:
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


def get_player_gamelog(player_id, season: str = "2025") -> FetchResult:
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

def get_espn_scoreboard(dates: Optional[str] = None) -> FetchResult:
    """dates format YYYYMMDD. Omit for today's slate."""
    params = {"dates": dates} if dates else None
    return _get_json(f"{ESPN_SITE_BASE}/scoreboard", params=params, cache_key=f"espn_scoreboard_{dates}")


def get_espn_team_schedule(espn_team_abbr: str, season: Optional[str] = None) -> FetchResult:
    params = {"season": season} if season else None
    return _get_json(
        f"{ESPN_SITE_BASE}/teams/{espn_team_abbr}/schedule",
        params=params,
        cache_key=f"espn_schedule_{espn_team_abbr}_{season}",
    )


def get_espn_injuries() -> FetchResult:
    """League-wide injury report. ESPN's injuries payload is nested per team."""
    return _get_json(f"{ESPN_SITE_BASE}/injuries", cache_key="espn_injuries")


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


def get_espn_player_gamelog(athlete_id) -> FetchResult:
    """Fallback player game log via ESPN's (undocumented) gamelog endpoint,
    used when stats.wnba.com is unreachable. Best-effort: returns
    FetchResult.fail rather than raising if the response doesn't match the
    expected shape, so a schema change here degrades safely to manual entry.
    """
    res = _get_json(
        f"{ESPN_COMMON_BASE}/athletes/{athlete_id}/gamelog",
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
                col_index = {}
                for i, label in enumerate(labels):
                    mapped = _ESPN_STAT_LABEL_MAP.get(str(label).upper())
                    if mapped:
                        col_index[mapped] = i
                for event in category.get("events", []):
                    event_id = event.get("eventId") or event.get("id")
                    stats = event.get("stats", [])
                    meta = events_meta.get(str(event_id), {}) if isinstance(events_meta, dict) else {}
                    opponent = (meta.get("opponent") or {}).get("abbreviation", "")
                    at_vs = meta.get("atVs", "vs")
                    row = {
                        "GAME_DATE": meta.get("gameDate"),
                        "MATCHUP": f"{at_vs} {opponent}".strip(),
                    }
                    for col, idx in col_index.items():
                        if idx < len(stats):
                            row[col] = stats[idx]
                    rows.append(row)
        if not rows:
            return FetchResult.fail("ESPN gamelog returned no parsable rows")
        return FetchResult.ok(rows)
    except (AttributeError, TypeError, KeyError, IndexError) as exc:
        return FetchResult.fail(f"unexpected ESPN gamelog shape: {exc}")


def get_espn_team_roster(espn_team_abbr: str) -> FetchResult:
    return _get_json(
        f"{ESPN_SITE_BASE}/teams/{espn_team_abbr}/roster",
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
