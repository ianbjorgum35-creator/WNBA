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
               cache_key: Optional[str] = None) -> FetchResult:
    if cache_key is not None:
        cached = _cache_get(cache_key)
        if cached is not None:
            return FetchResult.ok(cached, message="from cache")
    try:
        resp = requests.get(
            url, params=params, headers=headers or DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        payload = resp.json()
    except requests.exceptions.RequestException as exc:
        return FetchResult.fail(f"request failed: {exc}")
    except ValueError as exc:
        return FetchResult.fail(f"bad JSON response: {exc}")
    if cache_key is not None:
        _cache_set(cache_key, payload)
    return FetchResult.ok(payload)


# --------------------------------------------------------------------------
# stats.wnba.com -- player game logs, team dashboards
# --------------------------------------------------------------------------

def _stats_endpoint(endpoint: str, params: dict, cache_key: str) -> FetchResult:
    return _get_json(f"{WNBA_STATS_BASE}/{endpoint}", params=params, cache_key=cache_key)


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
