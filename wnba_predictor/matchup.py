"""Game-environment and matchup factors: pace/defense ranks, opponent-allowed-stat rank, H2H, rest."""

from datetime import date
from typing import Optional

import pandas as pd

from . import stats_engine
from .odds import implied_team_totals

N_TEAMS = 15


def team_pace_and_defense(team_ranks: dict, opponent_team: str) -> dict:
    """Look up the opponent's pace/defense ranks from data_sources.get_team_ranks output."""
    entry = team_ranks.get(opponent_team)
    if entry is None:
        return {"pace_rank": None, "def_rating_rank": None, "off_rating_rank": None, "note": "team not found"}
    return entry


def team_stat_allowed_rank(stat_ranks: dict, opponent_team: str) -> dict:
    """Look up the opponent's rank from data_sources.get_espn_stat_allowed_ranks
    output -- the DvP replacement (team-level, stat-specific, not
    position-specific: "how much of this stat does this opponent give up
    per game, ranked league-wide," rank 1 = allows the most)."""
    entry = stat_ranks.get(opponent_team)
    if entry is None:
        return {"stat_allowed_rank": None, "stat_allowed_avg": None, "note": "team not found"}
    return entry


def _extract_opponent_from_matchup(matchup: str, own_team_abbr: str) -> Optional[str]:
    if not isinstance(matchup, str):
        return None
    parts = matchup.replace("vs.", "@").split("@")
    for p in parts:
        p = p.strip()
        if p and p != own_team_abbr:
            return p
    return None


def h2h_hit_rate(player_gamelog: pd.DataFrame, own_team_abbr: str, opponent_team_abbr: str,
                  prop_key: str, line: float, direction: str) -> dict:
    if player_gamelog.empty or "MATCHUP" not in player_gamelog.columns:
        return {"h2h_hit_rate": None, "h2h_games": 0}
    opp_col = player_gamelog["MATCHUP"].apply(
        lambda m: _extract_opponent_from_matchup(m, own_team_abbr)
    )
    subset = player_gamelog[opp_col == opponent_team_abbr]
    if subset.empty:
        return {"h2h_hit_rate": None, "h2h_games": 0}
    values = stats_engine.stat_series(subset, prop_key)
    hit_rate = stats_engine._hit_rate(values, line, direction)
    return {"h2h_hit_rate": hit_rate, "h2h_games": int(values.dropna().shape[0]), "h2h_avg": float(values.mean())}


def rest_and_b2b(team_game_dates: list, upcoming_game_date: date) -> dict:
    prior = [d for d in team_game_dates if d < upcoming_game_date]
    if not prior:
        return {"rest_days": None, "is_b2b": None}
    last_game = max(prior)
    rest_days = (upcoming_game_date - last_game).days - 1
    return {"rest_days": rest_days, "is_b2b": rest_days <= 0}


def game_environment(game_total: float, spread: float, is_home: bool) -> dict:
    """Spread is entered from the *player's team* perspective (negative = favorite)."""
    team_implied, opp_implied = implied_team_totals(game_total, spread)
    return {
        "team_implied_total": team_implied,
        "opponent_implied_total": opp_implied,
        "is_home": is_home,
        "is_favorite": spread < 0,
    }
