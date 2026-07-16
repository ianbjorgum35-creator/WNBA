"""Tests for the network-adjacent parts of data_sources.py that don't
require live network access: retry/backoff behavior and the ESPN gamelog
fallback parser (built against an assumed schema for the endpoint, since it
could not be verified against live traffic in the build sandbox).
"""

import os
import sys
from datetime import date, timedelta

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wnba_predictor import data_sources


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


def test_get_json_retries_on_timeout_then_succeeds(monkeypatch):
    monkeypatch.setattr(data_sources.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.Timeout("simulated timeout")
        return _FakeResponse({"ok": True})

    monkeypatch.setattr(requests, "get", fake_get)
    result = data_sources._get_json("https://example.invalid/x", retries=2)
    assert result.success is True
    assert calls["n"] == 2


def test_get_json_fails_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(data_sources.time, "sleep", lambda *_: None)

    def fake_get(url, params=None, headers=None, timeout=None):
        raise requests.exceptions.Timeout("simulated timeout")

    monkeypatch.setattr(requests, "get", fake_get)
    result = data_sources._get_json("https://example.invalid/x", retries=3)
    assert result.success is False
    assert "timed out" in result.message


def test_get_json_does_not_retry_on_bad_json(monkeypatch):
    calls = {"n": 0}

    class _BadJsonResponse:
        def raise_for_status(self):
            pass

        def json(self):
            raise ValueError("not json")

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        return _BadJsonResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    result = data_sources._get_json("https://example.invalid/x", retries=3)
    assert result.success is False
    assert calls["n"] == 1  # no point retrying a malformed response


def test_espn_player_id_matches_flat_roster(monkeypatch):
    fake_roster = {
        "athletes": [
            {"id": "111", "displayName": "Jane Starter"},
            {"id": "222", "displayName": "Sam Bench"},
        ]
    }
    monkeypatch.setattr(
        data_sources, "get_espn_team_roster",
        lambda abbr: data_sources.FetchResult.ok(fake_roster),
    )
    result = data_sources.get_espn_player_id("Jane Starter", "IND")
    assert result.success is True
    assert result.data == "111"


def test_espn_player_id_matches_grouped_roster(monkeypatch):
    fake_roster = {
        "athletes": [
            {"position": "Guards", "items": [{"id": "333", "displayName": "Guard One"}]},
            {"position": "Forwards", "items": [{"id": "444", "displayName": "Forward One"}]},
        ]
    }
    monkeypatch.setattr(
        data_sources, "get_espn_team_roster",
        lambda abbr: data_sources.FetchResult.ok(fake_roster),
    )
    result = data_sources.get_espn_player_id("Forward One", "IND")
    assert result.success is True
    assert result.data == "444"


def test_espn_player_gamelog_parses_expected_shape(monkeypatch):
    fake_payload = {
        "events": {
            "401": {"gameDate": "2025-06-01", "atVs": "vs", "opponent": {"abbreviation": "CHI"}},
            "402": {"gameDate": "2025-06-03", "atVs": "@", "opponent": {"abbreviation": "NYL"}},
        },
        "seasonTypes": [
            {
                "displayName": "Regular Season",
                "categories": [
                    {
                        "labels": ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TO", "3PT"],
                        "events": [
                            {"eventId": "401", "stats": ["32", "18", "6", "4", "1", "0", "2", "3"]},
                            {"eventId": "402", "stats": ["29", "14", "5", "3", "2", "1", "1", "1"]},
                        ],
                    }
                ],
            }
        ],
    }
    monkeypatch.setattr(
        data_sources, "_get_json",
        lambda *a, **k: data_sources.FetchResult.ok(fake_payload),
    )
    result = data_sources.get_espn_player_gamelog("999")
    assert result.success is True
    assert len(result.data) == 2
    row = result.data[0]
    assert row["MATCHUP"] == "vs CHI"
    assert row["PTS"] == "18"
    assert row["FG3M"] == "3"


def test_espn_player_gamelog_fails_soft_on_unexpected_shape(monkeypatch):
    monkeypatch.setattr(
        data_sources, "_get_json",
        lambda *a, **k: data_sources.FetchResult.ok({"totally": "different shape"}),
    )
    result = data_sources.get_espn_player_gamelog("999")
    assert result.success is False


def test_espn_player_gamelog_parses_name_value_stats_shape(monkeypatch):
    # Per-event stats as {"name"/"value"} dicts rather than a parallel array
    # aligned to category labels -- the shape confirmed live for ESPN's
    # per-team /statistics endpoint, and plausible here too.
    fake_payload = {
        "events": {
            "401": {"gameDate": "2025-06-01", "atVs": "vs", "opponent": {"abbreviation": "CHI"}},
        },
        "seasonTypes": [{
            "categories": [{
                "labels": [],  # deliberately empty/unused for this shape
                "events": [
                    {"eventId": "401", "stats": [
                        {"name": "PTS", "value": "18"},
                        {"name": "REB", "value": "6"},
                    ]},
                ],
            }],
        }],
    }
    monkeypatch.setattr(
        data_sources, "_get_json",
        lambda *a, **k: data_sources.FetchResult.ok(fake_payload),
    )
    result = data_sources.get_espn_player_gamelog("999")
    assert result.success is True
    row = result.data[0]
    assert row["PTS"] == "18"
    assert row["REB"] == "6"


def test_espn_player_gamelog_fails_when_rows_parse_but_no_stats_extracted(monkeypatch):
    # This is exactly the real-world failure mode observed live: dates and
    # opponents parse fine (row count looks right, UI reports "N games
    # loaded"), but the label/stat shape didn't match anything in
    # _ESPN_STAT_LABEL_MAP, so every stat column would silently be empty.
    # That must surface as a failure, not a false "success".
    fake_payload = {
        "events": {"401": {"gameDate": "2025-06-01", "atVs": "vs", "opponent": {"abbreviation": "CHI"}}},
        "seasonTypes": [{
            "categories": [{
                "labels": ["SomeUnrecognizedLabel"],
                "events": [{"eventId": "401", "stats": ["99"]}],
            }],
        }],
    }
    monkeypatch.setattr(
        data_sources, "_get_json",
        lambda *a, **k: data_sources.FetchResult.ok(fake_payload),
    )
    result = data_sources.get_espn_player_gamelog("999")
    assert result.success is False
    assert "no recognizable stat columns" in result.message


def test_dump_espn_gamelog_shape_summarizes_structure(monkeypatch):
    fake_payload = {
        "events": {"401": {"gameDate": "2025-06-01", "opponent": {"abbreviation": "CHI"}}},
        "seasonTypes": [{
            "categories": [{
                "labels": ["MIN", "PTS"],
                "events": [{"eventId": "401", "stats": ["32", "18"]}],
            }],
        }],
    }
    monkeypatch.setattr(
        data_sources, "_espn_endpoint",
        lambda url, params, cache_key: data_sources.FetchResult.ok(fake_payload),
    )
    summary = data_sources.dump_espn_gamelog_shape("999")
    assert "seasonTypes" in summary["top_level_keys"]
    assert summary["category_labels"] == ["MIN", "PTS"]
    assert summary["sample_event"]["eventId"] == "401"


def test_dump_espn_gamelog_shape_reports_error(monkeypatch):
    monkeypatch.setattr(
        data_sources, "_espn_endpoint",
        lambda url, params, cache_key: data_sources.FetchResult.fail("timed out"),
    )
    summary = data_sources.dump_espn_gamelog_shape("999")
    assert "timed out" in summary["error"]


def test_dump_espn_gamelog_shape_surfaces_top_level_labels(monkeypatch):
    # This is exactly what the previous dump missed: labels living at the
    # payload's top level rather than nested under the category, which is
    # what caused category_labels to report None against live traffic.
    fake_payload = {
        "names": ["MIN", "PTS"],
        "events": {"401": {"gameDate": "2025-06-01", "opponent": {"abbreviation": "CHI"}}},
        "seasonTypes": [{"categories": [{"events": [{"eventId": "401", "stats": ["32", "18"]}]}]}],
    }
    monkeypatch.setattr(
        data_sources, "_espn_endpoint",
        lambda url, params, cache_key: data_sources.FetchResult.ok(fake_payload),
    )
    summary = data_sources.dump_espn_gamelog_shape("999")
    assert summary["top_level_names"] == ["MIN", "PTS"]
    assert summary["category_labels"] is None  # confirms category-level really is empty here


def test_parse_stat_value_extracts_made_count_from_combo_string():
    assert data_sources._parse_stat_value("1-6") == "1"
    assert data_sources._parse_stat_value("0-2") == "0"
    assert data_sources._parse_stat_value("31") == "31"
    assert data_sources._parse_stat_value(None) is None
    assert data_sources._parse_stat_value("-5") == "-5"  # not a combo -- leave a real negative alone


def test_espn_player_gamelog_falls_back_to_top_level_labels_and_parses_combo_stats(monkeypatch):
    # Reproduces the exact real live payload shape: labels at the top level
    # (not nested under category), *both* a verbose "names" list ("minutes",
    # "points", ...) and a short "labels" list ("MIN", "PTS", ...) present
    # simultaneously (the short one must win, since that's what
    # _ESPN_STAT_LABEL_MAP is keyed on), and FG/3PT/FT stats formatted as
    # "made-attempted" combo strings that must be reduced to the made count.
    fake_payload = {
        "names": [
            "minutes", "points", "totalRebounds", "assists", "steals", "blocks", "turnovers",
            "fieldGoalsMade-fieldGoalsAttempted", "fieldGoalPct",
            "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "threePointPct",
            "freeThrowsMade-freeThrowsAttempted", "freeThrowPct", "fouls",
        ],
        "labels": ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TO", "FG", "FG%", "3PT", "3P%", "FT", "FT%", "PF"],
        "events": {
            "401857067": {
                "gameDate": "2026-07-14T23:00:00.000+00:00", "atVs": "@",
                "opponent": {"abbreviation": "TOR"},
            },
        },
        "seasonTypes": [{
            "categories": [{
                "displayName": "Regular Season",
                "events": [{
                    "eventId": "401857067",
                    "stats": ["28", "17", "10", "0", "1", "3", "1", "7-14", "50.0", "0-3", "0.0", "3-4", "75.0", "1"],
                }],
            }],
        }],
    }
    monkeypatch.setattr(
        data_sources, "_get_json",
        lambda *a, **k: data_sources.FetchResult.ok(fake_payload),
    )
    result = data_sources.get_espn_player_gamelog("999")
    assert result.success is True
    row = result.data[0]
    assert row["MIN"] == "28"
    assert row["PTS"] == "17"
    assert row["REB"] == "10"
    assert row["AST"] == "0"
    assert row["STL"] == "1"
    assert row["BLK"] == "3"
    assert row["TOV"] == "1"
    assert row["FG3M"] == "0"  # from the "0-3" 3PT combo -- made count only
    assert row["FG3M"] == "0"  # from the "0-2" 3PT combo -- made count only


def _fake_espn_teams_payload():
    return {
        "sports": [{
            "leagues": [{
                "teams": [
                    {"team": {"id": "1", "abbreviation": "IND", "displayName": "Indiana Fever"}},
                    {"team": {"id": "2", "abbreviation": "CONN", "displayName": "Connecticut Sun"}},
                ]
            }]
        }]
    }


def test_get_espn_teams_parses_expected_shape(monkeypatch):
    monkeypatch.setattr(
        data_sources, "_get_json",
        lambda *a, **k: data_sources.FetchResult.ok(_fake_espn_teams_payload()),
    )
    result = data_sources.get_espn_teams()
    assert result.success is True
    assert len(result.data) == 2
    assert result.data[0]["abbreviation"] == "IND"


def test_get_espn_teams_fails_soft_on_unexpected_shape(monkeypatch):
    monkeypatch.setattr(
        data_sources, "_get_json",
        lambda *a, **k: data_sources.FetchResult.ok({"nope": True}),
    )
    result = data_sources.get_espn_teams()
    assert result.success is False


def test_resolve_espn_team_abbr_finds_mismatched_abbreviation(monkeypatch):
    # This is exactly the case the resolver exists for: our config abbreviates
    # Connecticut as "CON", but ESPN's own abbreviation for it is "CONN".
    monkeypatch.setattr(
        data_sources, "get_espn_teams",
        lambda: data_sources.FetchResult.ok([
            {"displayName": "Indiana Fever", "abbreviation": "IND"},
            {"displayName": "Connecticut Sun", "abbreviation": "CONN"},
        ]),
    )
    result = data_sources.resolve_espn_team_abbr("Connecticut Sun")
    assert result.success is True
    assert result.data == "CONN"


def test_resolve_espn_team_abbr_no_match(monkeypatch):
    monkeypatch.setattr(
        data_sources, "get_espn_teams",
        lambda: data_sources.FetchResult.ok([{"displayName": "Indiana Fever", "abbreviation": "IND"}]),
    )
    result = data_sources.resolve_espn_team_abbr("Nonexistent Team")
    assert result.success is False


def test_diagnostics_reports_per_check_status(monkeypatch):
    monkeypatch.setattr(data_sources, "get_team_ranks", lambda: data_sources.FetchResult.ok({"x": 1}))
    monkeypatch.setattr(data_sources, "get_espn_scoreboard", lambda: data_sources.FetchResult.fail("boom"))
    monkeypatch.setattr(data_sources, "get_espn_teams", lambda: data_sources.FetchResult.ok([]))
    monkeypatch.setattr(data_sources, "resolve_espn_team_abbr", lambda name: data_sources.FetchResult.ok("IND"))
    monkeypatch.setattr(data_sources, "get_espn_team_roster", lambda abbr: data_sources.FetchResult.ok({}))
    monkeypatch.setattr(data_sources, "get_espn_team_schedule_parsed", lambda abbr, season=None: data_sources.FetchResult.ok([]))
    monkeypatch.setattr(data_sources, "get_espn_team_ranks_fallback", lambda: data_sources.FetchResult.ok({}))
    monkeypatch.setattr(data_sources, "get_player_id", lambda name: data_sources.FetchResult.fail("not found"))
    monkeypatch.setattr(data_sources, "get_espn_player_id", lambda name, abbr: data_sources.FetchResult.ok("123"))
    monkeypatch.setattr(data_sources, "get_espn_player_gamelog", lambda athlete_id: data_sources.FetchResult.ok([]))

    report = data_sources.diagnostics(player_name="Test Player", player_team_name="Indiana Fever")

    assert report["stats_wnba_reachability (team ranks)"]["ok"] is True
    assert report["espn_reachability (scoreboard)"]["ok"] is False
    assert report["espn_resolve_team_abbr"]["ok"] is True
    assert report["espn_team_roster"]["ok"] is True
    assert report["espn_team_ranks_fallback (standings/schedule-scores)"]["ok"] is True
    assert report["stats_wnba_player_id"]["ok"] is False
    assert report["espn_player_id"]["ok"] is True
    assert report["espn_player_gamelog"]["ok"] is True


def _fake_event(game_date, own_abbr, opp_abbr, is_home, completed):
    return {
        "date": f"{game_date.isoformat()}T23:00Z",
        "competitions": [{
            "date": f"{game_date.isoformat()}T23:00Z",
            "competitors": [
                {"team": {"abbreviation": own_abbr}, "homeAway": "home" if is_home else "away"},
                {"team": {"abbreviation": opp_abbr}, "homeAway": "away" if is_home else "home"},
            ],
            "status": {"type": {"completed": completed}},
        }],
    }


def test_get_espn_team_schedule_parsed_parses_expected_shape(monkeypatch):
    d1, d2 = date(2026, 6, 1), date(2026, 6, 3)
    fake_payload = {"events": [
        _fake_event(d1, "IND", "CHI", True, True),
        _fake_event(d2, "IND", "NYL", False, True),
    ]}
    monkeypatch.setattr(
        data_sources, "get_espn_team_schedule",
        lambda abbr, season=None: data_sources.FetchResult.ok(fake_payload),
    )
    result = data_sources.get_espn_team_schedule_parsed("IND")
    assert result.success is True
    assert len(result.data) == 2
    assert result.data[0]["date"] == d1
    assert result.data[0]["is_home"] is True
    assert result.data[0]["opponent_abbr"] == "CHI"
    assert result.data[1]["opponent_abbr"] == "NYL"


def test_get_espn_team_schedule_parsed_fails_soft_on_unexpected_shape(monkeypatch):
    monkeypatch.setattr(
        data_sources, "get_espn_team_schedule",
        lambda abbr, season=None: data_sources.FetchResult.ok({"nope": True}),
    )
    result = data_sources.get_espn_team_schedule_parsed("IND")
    assert result.success is False


def test_find_scheduled_game_prefers_upcoming():
    today = date.today()
    games = [
        {"date": today - timedelta(days=10), "opponent_abbr": "CHI", "is_home": True, "completed": True},
        {"date": today + timedelta(days=5), "opponent_abbr": "CHI", "is_home": False, "completed": False},
        {"date": today + timedelta(days=2), "opponent_abbr": "NYL", "is_home": True, "completed": False},
    ]
    result = data_sources.find_scheduled_game(games, "CHI")
    assert result["date"] == today + timedelta(days=5)


def test_find_scheduled_game_falls_back_to_most_recent_past():
    today = date.today()
    games = [
        {"date": today - timedelta(days=20), "opponent_abbr": "CHI", "is_home": True, "completed": True},
        {"date": today - timedelta(days=5), "opponent_abbr": "CHI", "is_home": False, "completed": True},
    ]
    result = data_sources.find_scheduled_game(games, "CHI")
    assert result["date"] == today - timedelta(days=5)


def test_find_scheduled_game_no_match_returns_none():
    assert data_sources.find_scheduled_game([{"date": date.today(), "opponent_abbr": "NYL"}], "CHI") is None


def test_game_context_computes_rest_days_and_home_away(monkeypatch):
    today = date.today()
    games = [
        {"date": today - timedelta(days=3), "opponent_abbr": "NYL", "is_home": True, "completed": True},
        {"date": today + timedelta(days=1), "opponent_abbr": "CHI", "is_home": False, "completed": False},
    ]
    monkeypatch.setattr(
        data_sources, "get_espn_team_schedule_parsed",
        lambda abbr, season=None: data_sources.FetchResult.ok(games),
    )
    result = data_sources.game_context("IND", "CHI")
    assert result.success is True
    assert result.data["is_home"] is False
    assert result.data["rest_days"] == 3  # 1 day before target minus the prior game date, exclusive
    assert result.data["is_b2b"] is False


def test_game_context_flags_back_to_back(monkeypatch):
    today = date.today()
    games = [
        {"date": today, "opponent_abbr": "NYL", "is_home": True, "completed": True},
        {"date": today + timedelta(days=1), "opponent_abbr": "CHI", "is_home": False, "completed": False},
    ]
    monkeypatch.setattr(
        data_sources, "get_espn_team_schedule_parsed",
        lambda abbr, season=None: data_sources.FetchResult.ok(games),
    )
    result = data_sources.game_context("IND", "CHI")
    assert result.data["is_b2b"] is True


def _fake_standings_payload():
    def entry(name, points_for, points_against, games_played):
        return {
            "team": {"displayName": name},
            "stats": [
                {"name": "pointsFor", "value": points_for},
                {"name": "pointsAgainst", "value": points_against},
                {"name": "gamesPlayed", "value": games_played},
            ],
        }

    return {
        "children": [{
            "standings": {
                "entries": [
                    entry("Indiana Fever", 1800, 1700, 20),   # 90.0 for / 85.0 against
                    entry("Chicago Sky", 1600, 1750, 20),     # 80.0 for / 87.5 against
                    entry("New York Liberty", 1900, 1600, 20),  # 95.0 for / 80.0 against
                ]
            }
        }]
    }


def test_get_espn_team_ranks_fallback_parses_and_ranks(monkeypatch):
    monkeypatch.setattr(
        data_sources, "get_espn_standings",
        lambda: data_sources.FetchResult.ok(_fake_standings_payload()),
    )
    result = data_sources.get_espn_team_ranks_fallback()
    assert result.success is True
    ranks = result.data
    # New York allows fewest points (80.0) -> best defense -> rank 1
    assert ranks["New York Liberty"]["def_rating_rank"] == 1
    # Indiana allows the most (85.0) among these three -> worse than NY, better than Chicago
    assert ranks["Indiana Fever"]["def_rating_rank"] == 2
    assert ranks["Chicago Sky"]["def_rating_rank"] == 3
    for team_ranks in ranks.values():
        assert team_ranks["source"] == "espn_standings_proxy"
        assert team_ranks["pace_rank"] is not None


def test_flatten_named_stats_finds_nested_values():
    payload = {
        "splits": [{"categories": [{"stats": [
            {"name": "avgPointsFor", "value": 82.5},
            {"name": "avgPointsAgainst", "value": 78.1},
            {"name": "teamName", "value": "not a number"},  # should be ignored
        ]}]}]
    }
    flat = data_sources._flatten_named_stats(payload)
    assert flat["avgPointsFor"] == 82.5
    assert flat["avgPointsAgainst"] == 78.1
    assert "teamName" not in flat


def test_extract_points_for_against_normalizes_season_totals():
    # season totals (>200) should be divided down to per-game using gamesPlayed
    stat_map = {"pointsFor": 1800, "pointsAgainst": 1700, "gamesPlayed": 20}
    points_for, points_against = data_sources._extract_points_for_against(stat_map)
    assert points_for == 90.0
    assert points_against == 85.0


def test_extract_points_for_against_leaves_per_game_values_alone():
    stat_map = {"avgPoints": 82.5, "avgPointsAllowed": 78.1}
    points_for, points_against = data_sources._extract_points_for_against(stat_map)
    assert points_for == 82.5
    assert points_against == 78.1


def test_get_espn_team_ranks_fallback_falls_through_to_schedule_scores(monkeypatch):
    # Standings has no scoring stats at all (just W-L-PCT, the real-world
    # failure mode confirmed live for this league) -- should fall through to
    # computing points-for/against from each team's own schedule results.
    monkeypatch.setattr(
        data_sources, "get_espn_standings",
        lambda: data_sources.FetchResult.ok({
            "children": [{"standings": {"entries": [
                {"team": {"displayName": "Indiana Fever"},
                 "stats": [{"name": "wins", "value": 10}, {"name": "losses", "value": 5}]},
            ]}}]
        }),
    )
    monkeypatch.setattr(
        data_sources, "get_espn_teams",
        lambda: data_sources.FetchResult.ok([
            {"abbreviation": "IND", "displayName": "Indiana Fever"},
            {"abbreviation": "CHI", "displayName": "Chicago Sky"},
        ]),
    )

    def fake_schedule_parsed(abbr, season=None):
        if abbr == "IND":
            games = [
                {"completed": True, "own_score": 92.0, "opponent_score": 85.0},
                {"completed": True, "own_score": 88.0, "opponent_score": 85.0},
            ]
        else:
            games = [
                {"completed": True, "own_score": 80.0, "opponent_score": 87.5},
                {"completed": False, "own_score": None, "opponent_score": None},
            ]
        return data_sources.FetchResult.ok(games)

    monkeypatch.setattr(data_sources, "get_espn_team_schedule_parsed", fake_schedule_parsed)

    result = data_sources.get_espn_team_ranks_fallback()
    assert result.success is True
    assert result.data["Indiana Fever"]["source"] == "espn_schedule_scores_proxy"
    assert result.data["Indiana Fever"]["def_rating_rank"] == 1  # 85.0 allowed < Chicago's 87.5


def test_get_espn_team_ranks_fallback_fails_soft_when_both_tiers_empty(monkeypatch):
    monkeypatch.setattr(
        data_sources, "get_espn_standings",
        lambda: data_sources.FetchResult.ok({"nope": True}),
    )
    monkeypatch.setattr(
        data_sources, "get_espn_teams",
        lambda: data_sources.FetchResult.fail("also broken"),
    )
    result = data_sources.get_espn_team_ranks_fallback()
    assert result.success is False


def test_get_espn_team_ranks_fallback_falls_through_on_standings_network_failure(monkeypatch):
    monkeypatch.setattr(
        data_sources, "get_espn_standings",
        lambda: data_sources.FetchResult.fail("timed out"),
    )
    monkeypatch.setattr(
        data_sources, "get_espn_teams",
        lambda: data_sources.FetchResult.fail("also timed out"),
    )
    result = data_sources.get_espn_team_ranks_fallback()
    assert result.success is False
    assert "no usable points-for/against" in result.message


def test_parse_espn_score_handles_dict_and_plain_shapes():
    assert data_sources._parse_espn_score({"value": 82.0}) == 82.0
    assert data_sources._parse_espn_score({"displayValue": "79"}) == 79.0
    assert data_sources._parse_espn_score("85") == 85.0
    assert data_sources._parse_espn_score(None) is None
    assert data_sources._parse_espn_score({"value": None, "displayValue": "not a number"}) is None


def test_get_espn_team_schedule_parsed_captures_scores(monkeypatch):
    fake_payload = {"events": [
        {
            "date": "2026-06-01T23:00Z",
            "competitions": [{
                "date": "2026-06-01T23:00Z",
                "competitors": [
                    {"team": {"abbreviation": "IND"}, "homeAway": "home", "score": {"value": 92.0}},
                    {"team": {"abbreviation": "CHI"}, "homeAway": "away", "score": {"value": 85.0}},
                ],
                "status": {"type": {"completed": True}},
            }],
        },
    ]}
    monkeypatch.setattr(
        data_sources, "get_espn_team_schedule",
        lambda abbr, season=None: data_sources.FetchResult.ok(fake_payload),
    )
    result = data_sources.get_espn_team_schedule_parsed("IND")
    assert result.success is True
    game = result.data[0]
    assert game["own_score"] == 92.0
    assert game["opponent_score"] == 85.0


def test_team_rows_from_schedule_scores_averages_completed_games(monkeypatch):
    monkeypatch.setattr(
        data_sources, "get_espn_teams",
        lambda: data_sources.FetchResult.ok([{"abbreviation": "IND", "displayName": "Indiana Fever"}]),
    )
    monkeypatch.setattr(
        data_sources, "get_espn_team_schedule_parsed",
        lambda abbr, season=None: data_sources.FetchResult.ok([
            {"completed": True, "own_score": 92.0, "opponent_score": 85.0},
            {"completed": True, "own_score": 88.0, "opponent_score": 81.0},
            {"completed": False, "own_score": None, "opponent_score": None},  # not yet played
        ]),
    )
    rows = data_sources._team_rows_from_schedule_scores()
    assert len(rows) == 1
    assert rows[0]["points_for_pg"] == 90.0
    assert rows[0]["points_against_pg"] == 83.0


def test_team_rows_from_schedule_scores_skips_teams_with_no_completed_games(monkeypatch):
    monkeypatch.setattr(
        data_sources, "get_espn_teams",
        lambda: data_sources.FetchResult.ok([{"abbreviation": "IND", "displayName": "Indiana Fever"}]),
    )
    monkeypatch.setattr(
        data_sources, "get_espn_team_schedule_parsed",
        lambda abbr, season=None: data_sources.FetchResult.ok([
            {"completed": False, "own_score": None, "opponent_score": None},
        ]),
    )
    rows = data_sources._team_rows_from_schedule_scores()
    assert rows == []
