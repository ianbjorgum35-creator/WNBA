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
    monkeypatch.setattr(data_sources, "get_player_id", lambda name: data_sources.FetchResult.fail("not found"))
    monkeypatch.setattr(data_sources, "get_espn_player_id", lambda name, abbr: data_sources.FetchResult.ok("123"))

    report = data_sources.diagnostics(player_name="Test Player", player_team_name="Indiana Fever")

    assert report["stats_wnba_reachability (team ranks)"]["ok"] is True
    assert report["espn_reachability (scoreboard)"]["ok"] is False
    assert report["espn_resolve_team_abbr"]["ok"] is True
    assert report["espn_team_roster"]["ok"] is True
    assert report["stats_wnba_player_id"]["ok"] is False
    assert report["espn_player_id"]["ok"] is True


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
