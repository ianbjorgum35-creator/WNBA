"""Tests for the network-adjacent parts of data_sources.py that don't
require live network access: retry/backoff behavior and the ESPN gamelog
fallback parser (built against an assumed schema for the endpoint, since it
could not be verified against live traffic in the build sandbox).
"""

import os
import sys

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
