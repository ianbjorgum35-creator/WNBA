"""Sanity tests using synthetic data -- no network calls.

These exercise odds math, the stats/matchup engines, the full projection
pipeline, and the CLV logging + learning loop end-to-end so the notebook's
logic is verified even though the live data-fetch endpoints can't be
network-tested in this environment.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wnba_predictor import clv, config, matchup, odds, projection, stats_engine


def test_implied_prob_roundtrip():
    for american in [-110, -250, +120, +350, -105]:
        p = odds.implied_prob_from_american(american)
        assert 0 < p < 1
        back = odds.american_from_implied_prob(p)
        # rounding tolerance
        assert abs(back - american) <= 2


def test_devig_two_way_sums_to_one():
    p_over, p_under = odds.devig_two_way(-115, -105)
    assert abs((p_over + p_under) - 1.0) < 1e-9
    assert p_over > 0 and p_under > 0


def test_edge_and_clv():
    assert odds.edge(0.55, 0.50) == pytest.approx(0.05)
    assert odds.clv_percent(0.50, 0.55) == pytest.approx(5.0)


def test_implied_team_totals():
    team, opp = odds.implied_team_totals(game_total=160, spread=-4)
    assert team == pytest.approx(82.0)
    assert opp == pytest.approx(78.0)
    assert team + opp == pytest.approx(160.0)


def _synthetic_gamelog(n=25, mean_pts=18, mean_min=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-05-01", periods=n, freq="3D")
    pts = rng.normal(mean_pts, 4, n).clip(0)
    reb = rng.normal(5, 2, n).clip(0)
    ast = rng.normal(4, 1.5, n).clip(0)
    minutes = rng.normal(mean_min, 3, n).clip(10, 40)
    rows = []
    for i in range(n):
        rows.append({
            "GAME_DATE": dates[i].strftime("%Y-%m-%d"),
            "MATCHUP": "IND vs. CHI" if i % 2 == 0 else "IND @ CHI",
            "MIN": minutes[i],
            "PTS": pts[i],
            "REB": reb[i],
            "AST": ast[i],
            "STL": rng.integers(0, 3),
            "BLK": rng.integers(0, 2),
            "TOV": rng.integers(0, 4),
            "FG3M": rng.integers(0, 4),
        })
    return rows


def test_normalize_and_rolling_windows():
    raw = _synthetic_gamelog()
    df = stats_engine.normalize_gamelog(raw)
    assert len(df) == 25
    assert df["GAME_DATE"].is_monotonic_increasing

    rolling = stats_engine.rolling_windows(df, "PTS", line=17.5, direction="Over")
    assert rolling["season_games"] == 25
    assert rolling["l5_games"] == 5
    assert 0 <= rolling["l10_hit_rate"] <= 1
    assert rolling["season_avg"] > 0


def test_minutes_and_usage_profile():
    df = stats_engine.normalize_gamelog(_synthetic_gamelog())
    minutes = stats_engine.minutes_profile(df)
    assert minutes["rotation_role"] in ("Starter", "Rotation", "Bench / Deep Rotation")
    usage = stats_engine.usage_profile(df)
    assert usage["source"] == "unavailable"  # no USG_PCT column in synthetic data


def test_teammate_on_off_split_insufficient_sample():
    player_df = stats_engine.normalize_gamelog(_synthetic_gamelog(n=25, seed=1))
    teammate_df = stats_engine.normalize_gamelog(_synthetic_gamelog(n=25, seed=1))
    result = stats_engine.teammate_on_off_split(player_df, teammate_df, "PTS")
    # identical schedules => "without teammate" bucket is empty
    assert result["bump_available"] is False


def test_h2h_hit_rate():
    df = stats_engine.normalize_gamelog(_synthetic_gamelog())
    result = matchup.h2h_hit_rate(df, "IND", "CHI", "PTS", 17.5, "Over")
    assert result["h2h_games"] == len(df)
    assert 0 <= result["h2h_hit_rate"] <= 1


def test_game_environment():
    env = matchup.game_environment(game_total=165, spread=-3, is_home=True)
    assert env["is_favorite"] is True
    assert env["team_implied_total"] + env["opponent_implied_total"] == pytest.approx(165)


def test_run_prediction_end_to_end():
    df = stats_engine.normalize_gamelog(_synthetic_gamelog(mean_pts=18, mean_min=30))
    rolling = stats_engine.rolling_windows(df, "PTS", line=17.5, direction="Over")
    minutes = stats_engine.minutes_profile(df)
    usage = stats_engine.usage_profile(df)

    inputs = projection.ProjectionInputs(
        prop_key="PTS",
        direction="Over",
        line=17.5,
        rolling=rolling,
        minutes_profile=minutes,
        projected_minutes=31.0,
        usage_profile=usage,
        opponent_pace_rank=2,
        opponent_def_rating_rank=12,
        dvp_rank=3,
        game_total=168,
        spread=-2,
        is_home=True,
        rest_days=2,
        is_b2b=False,
        h2h={"h2h_hit_rate": 0.6},
        manual_matchup_adjustment=0.02,
        context_lean=0.0,
        lineup_confirmed=True,
    )
    result = projection.run_prediction(inputs)
    assert 0 < result.predicted_prob < 1
    assert result.projected_mean > 0
    assert result.confidence in ("Low", "Medium", "High")
    assert set(result.breakdown.keys()) >= {
        "season_avg", "l20_avg", "l10_avg", "l5_avg", "l20_hit_rate", "l10_hit_rate",
        "l5_hit_rate", "minutes_per_game", "projected_minutes", "usage_pct", "starter",
        "rotation_role", "opponent_pace_rank", "opponent_def_rating_rank", "dvp_rank",
        "game_total", "spread", "home_away", "rest_days", "back_to_back", "h2h_hit_rate",
        "manual_matchup_adjustment", "context_lean", "lineup_confirmed",
    }

    # Unconfirmed lineup should widen variance and pull probability toward 0.5
    inputs.lineup_confirmed = False
    result_unconfirmed = projection.run_prediction(inputs)
    assert result_unconfirmed.projected_std > result.projected_std


def test_poisson_prop_distribution():
    df = stats_engine.normalize_gamelog(_synthetic_gamelog())
    rolling = stats_engine.rolling_windows(df, "FG3M", line=1.5, direction="Over")
    minutes = stats_engine.minutes_profile(df)
    inputs = projection.ProjectionInputs(
        prop_key="FG3M", direction="Over", line=1.5,
        rolling=rolling, minutes_profile=minutes, projected_minutes=30.0,
    )
    result = projection.run_prediction(inputs)
    assert 0 < result.predicted_prob < 1


def test_clv_log_and_learning_roundtrip(tmp_path):
    log_path = str(tmp_path / "bet_log.csv")
    weights_path = str(tmp_path / "learned_weights.json")

    rng = np.random.default_rng(42)
    bet_ids = []
    for i in range(30):
        predicted_prob = float(rng.uniform(0.4, 0.75))
        record = {
            "player_name": "Test Player",
            "player_team": "Indiana Fever",
            "opponent_team": "Chicago Sky",
            "prop_type": "Points",
            "direction": "Over",
            "line": 17.5,
            "odds_open": -110,
            "implied_prob_open": odds.implied_prob_from_american(-110),
            "predicted_prob": predicted_prob,
            "raw_predicted_prob": predicted_prob,
            "edge": predicted_prob - odds.implied_prob_from_american(-110),
            "season_hit_rate": float(rng.uniform(0.3, 0.7)),
            "l20_hit_rate": float(rng.uniform(0.3, 0.7)),
            "l10_hit_rate": float(rng.uniform(0.3, 0.7)),
            "l5_hit_rate": float(rng.uniform(0.3, 0.7)),
            "blend_weights_json": json.dumps(config.DEFAULT_BLEND_WEIGHTS),
            "adjustments_json": "{}",
            "breakdown_json": "{}",
            "confidence": "Medium",
            "lineup_confirmed": True,
        }
        bet_id = clv.log_bet(record, path=log_path)
        bet_ids.append((bet_id, predicted_prob))

    for bet_id, p in bet_ids:
        rng2 = np.random.default_rng(hash(bet_id) % (2**32))
        hit_value = 20 if rng2.random() < p else 15  # line is 17.5
        clv.record_outcome(bet_id, actual_stat_value=hit_value, closing_odds=-115, path=log_path)

    report = clv.calibration_report(log_path)
    assert report["n_bets"] == 30
    assert 0 <= report["win_rate"] <= 1
    assert report["brier_score"] >= 0

    fit = clv.fit_recalibration(log_path, min_bets=20, weights_path=weights_path)
    assert fit["fitted"] is True
    assert os.path.exists(weights_path)

    calibrator = clv.load_calibrator(weights_path)
    calibrated = calibrator(0.6, {})
    assert 0 < calibrated < 1

    weight_update = clv.update_blend_weights(log_path, min_bets=20)
    assert weight_update["updated"] is True
    assert abs(sum(weight_update["weights"].values()) - 1.0) < 1e-6


def test_calibration_report_empty_log(tmp_path):
    log_path = str(tmp_path / "empty_log.csv")
    report = clv.calibration_report(log_path)
    assert report["n_bets"] == 0
